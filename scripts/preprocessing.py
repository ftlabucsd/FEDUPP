"""Utilities for loading and preprocessing FED3 session data from CSV exports."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Core data containers


@dataclass(frozen=True)
class SessionKey:
    """Identifier for a single session."""

    session_id: str
    mouse_id: str
    group: str
    session_type: str
    session_path: Path


@dataclass
class SessionData:
    """Container for a preprocessed FED3 session."""

    key: SessionKey
    raw: pd.DataFrame

    @property
    def session_minutes(self) -> float:
        if self.raw.empty:
            return 0.0
        delta = self.raw["Time"].iloc[-1] - self.raw["Time"].iloc[0]
        return delta.total_seconds() / 60


# ---------------------------------------------------------------------------
# Loading helpers


def load_group_map(group_map_path: str | os.PathLike) -> Dict[str, List[str]]:
    path = Path(group_map_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    groups = payload.get("groups", {})
    return {name: list(ids) for name, ids in groups.items()}


def discover_sessions(sample_root: str | os.PathLike) -> Dict[str, List[Path]]:
    base = Path(sample_root)
    session_map: Dict[str, List[Path]] = {}
    for mouse_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        csv_files = sorted(mouse_dir.glob("*.csv"))
        if not csv_files:
            continue
        session_map[mouse_dir.name] = csv_files
    return session_map


def infer_session_type(session_df: pd.DataFrame) -> Optional[str]:
    if "Session_type" not in session_df.columns:
        return None
    value = str(session_df["Session_type"].iloc[0])
    if "FR1" in value.upper():
        return "FR1"
    if "REV" in value.upper():
        return "REV"
    return None


def load_session_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_columns = {
        "MM:DD:YYYY hh:mm:ss",
        "Event",
        "Active_Poke",
        "Pellet_Count",
        "Left_Poke_Count",
        "Right_Poke_Count",
    }
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing in {csv_path.name}: {sorted(missing)}")

    df = df.rename(columns={
        "MM:DD:YYYY hh:mm:ss": "Time",
        "Retrieval_Time": "collect_time",
        "Cum_Sum": "Cum_Sum",
        "Percent_Correct": "Percent_Correct",
    })

    df = df.replace(
        {
            "LeftWithPellet": "Left",
            "LeftDuringDispense": "Left",
            "RightWithPellet": "Right",
            "RightDuringDispense": "Right",
        }
    )
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).reset_index(drop=True)

    df["collect_time"] = df.get("collect_time", 0).apply(convert_to_numeric)
    max_value = get_max_numeric(df["collect_time"].copy())
    df["collect_time"] = df["collect_time"].replace("Timed_out", max_value).fillna(0)
    df["collect_time"] = pd.to_numeric(df["collect_time"], errors="coerce").fillna(0)

    for poke_col in [
            "Left_Poke_Count", "Right_Poke_Count",
            "Left_Poke_Count_Adj", "Right_Poke_Count_Adj",
            "Pellet_Count", "Pellet_Count_Adj",
            "Pellet_Count.1", "Pellet_Count.3", "Pellet_Count.4"]:
        if poke_col in df.columns:
            df[poke_col] = pd.to_numeric(df[poke_col], errors="coerce").fillna(0)

    df = ensure_percent_correct(df)

    df = df.reset_index(drop=True)
    df["Time_passed"] = df["Time"] - df["Time"].iloc[0]
    return df


def motor_turn_summary(csv_path: Path, cutoff: int = 15) -> Tuple[int, float]:
    df = pd.read_csv(csv_path)
    if "Motor_Turns" not in df.columns:
        return 0, 0.0

    pellets = df[(df["Event"] == "Pellet") & (df["Motor_Turns"] > 1)]
    if pellets.empty:
        return 0, 0.0
    exceeding = (pellets["Motor_Turns"] > cutoff).sum()
    return int(exceeding), float(exceeding / len(pellets))


def build_session_catalog(
    sample_root: str | os.PathLike,
    group_map_path: str | os.PathLike,
) -> Tuple[Dict[str, SessionData], Dict[str, Dict[str, List[SessionKey]]]]:
    group_map = load_group_map(group_map_path)
    session_paths = discover_sessions(sample_root)

    session_store: Dict[str, SessionData] = {}
    groupings: Dict[str, Dict[str, List[SessionKey]]] = {}

    for group_name, mouse_ids in group_map.items():
        groupings[group_name] = {"FR1": [], "REV": [], "UNKNOWN": []}
        for mouse_id in mouse_ids:
            for csv_path in session_paths.get(mouse_id, []):
                try:
                    df = load_session_csv(csv_path)
                except Exception:
                    continue

                session_type = infer_session_type(df) or "UNKNOWN"
                session_id = f"{mouse_id}_{csv_path.stem}"
                key = SessionKey(
                    session_id=session_id,
                    mouse_id=mouse_id,
                    group=group_name,
                    session_type=session_type,
                    session_path=csv_path,
                )
                session_store[session_id] = SessionData(key=key, raw=df)

                if session_type not in groupings[group_name]:
                    groupings[group_name][session_type] = []
                groupings[group_name][session_type].append(key)

    return session_store, groupings


# ---------------------------------------------------------------------------
# Legacy helpers retained for downstream compatibility


def convert_to_numeric(value):
    if isinstance(value, str) and value.isnumeric():
        return pd.to_numeric(value)
    return value


def ensure_percent_correct(df: pd.DataFrame) -> pd.DataFrame:
    df = calculate_accuracy_by_row(df, convert_large=True)
    return df


def calculate_accuracy_by_row(df: pd.DataFrame, convert_large: bool = True):
    active_col = f"{df['Active_Poke'].iloc[0]}_Poke_Count"
    if active_col not in df:
        df["Percent_Correct"] = 0.0
        return df

    left = pd.to_numeric(df["Left_Poke_Count"], errors="coerce").fillna(0)
    right = pd.to_numeric(df["Right_Poke_Count"], errors="coerce").fillna(0)

    denom = left + right
    denom = denom.replace(0, np.nan)
    num = pd.to_numeric(df[active_col], errors="coerce").fillna(0)

    percent = num / denom
    percent = percent.fillna(0)
    if convert_large:
        percent *= 100

    df["Percent_Correct"] = percent
    return df


def get_max_numeric(series: pd.Series):
    numeric_values = pd.to_numeric(series, errors="coerce")
    return numeric_values.max(skipna=True)


def get_retrieval_time(csv_path: str | os.PathLike, day: int = 3):
    data = load_session_csv(Path(csv_path))
    data = data[data["Time_passed"] < timedelta(days=day)]

    times = data["collect_time"].tolist()
    pellet_times = [each for each in times if each != 0.0]
    pellet_times = list(map(float, pellet_times))
    pellet_times = [each for each in pellet_times if not math.isnan(each)]
    return pellet_times


# ---------------------------------------------------------------------------
# Convenience getters for external modules


@lru_cache(maxsize=1)
def session_cache(sample_root: str | os.PathLike, group_map_path: str | os.PathLike):
    return build_session_catalog(sample_root, group_map_path)


def get_group_sessions(
    sample_root: str | os.PathLike,
    group_map_path: str | os.PathLike,
    session_type: str,
    group_name: Optional[str] = None,
) -> Dict[str, List[SessionKey]]:
    _, groupings = session_cache(sample_root, group_map_path)
    if group_name:
        return {group_name: groupings.get(group_name, {}).get(session_type, [])}
    return {grp: info.get(session_type, []) for grp, info in groupings.items()}


def get_session_data(sample_root: str | os.PathLike, group_map_path: str | os.PathLike, session_id: str) -> SessionData:
    sessions, _ = session_cache(sample_root, group_map_path)
    return sessions[session_id]


def get_mouse_label(session: SessionData) -> str:
    return session.key.mouse_id