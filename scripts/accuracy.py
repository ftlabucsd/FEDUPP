"""Accuracy metrics and plots used throughout the FED3 analysis pipeline."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.preprocessing import SessionData
import numpy as np
from datetime import datetime, timedelta
from scripts.utils import palette

def read_and_record(session: SessionData, ending_corr: list, learned_time: list, acc_dict: dict) -> pd.DataFrame:
    """Extract accuracy metrics from a session and return the first 24.5 hours.

    Args:
        session (SessionData): Preprocessed session with ``Percent_Correct``.
        ending_corr (list): Mutable list that collects accuracy near 24.5 hours.
        learned_time (list): Mutable list that stores the learning milestone time.
        acc_dict (dict): Dictionary keyed by mouse ID that records accuracy
            near the end of the window.

    Returns:
        pd.DataFrame: Portion of the session up to the accuracy snapshot index.
    """
    df = session.raw.copy()
    target_time = timedelta(hours=24, minutes=30)
    df['time_diff'] = (df['Time_passed'] - target_time).abs()
    closest_accuracy = df.loc[df['time_diff'].idxmin(), 'Percent_Correct']
    cutoff_idx = df['time_diff'].idxmin()
    truncated = df.iloc[:cutoff_idx].copy()

    ending_corr.append(closest_accuracy)
    learned_time.append(find_learning_milestone(truncated))
    acc_dict[session.key.mouse_id] = closest_accuracy
    return truncated


def plot_cumulative_accuracy(
    groups: list,
    group_labels: list | None = None,
    bin_size_sec: int = 10,
    export_path: str | None = None,
):
    """Plot mean ± SEM cumulative accuracy for each cohort over 24 hours.

    Args:
        groups (list): For each cohort, a list of session DataFrames returned by
            ``read_and_record``.
        group_labels (list[str] | None): Optional names for the legend.
        bin_size_sec (int): Resampling interval in seconds.
        export_path (str | None): When provided, save the figure to this path.
    """
    if group_labels is None:
        group_labels = [f"G{idx+1}" for idx in range(len(groups))]

    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)

    for grp_idx, (group, label) in enumerate(zip(groups, group_labels)):
        binned_series = []

        for subj_idx, df in enumerate(group):
            ts = df.set_index('Time_passed')['Percent_Correct']
            resampled = (
                ts
                .resample(f"{bin_size_sec}s")
                .mean()
                .ffill()
            )
            hours = resampled.index.view('int64') / 1e9 / 3600
            resampled.index = hours
            resampled.name = f"{label}_{subj_idx}"
            binned_series.append(resampled)

        all_binned = pd.concat(binned_series, axis=1)
        cutoff = 24 * 3600 // bin_size_sec
        counts  = all_binned.count(axis=1).iloc[:cutoff]
        mean_pc = all_binned.mean(axis=1).iloc[:cutoff]
        std_pc  = all_binned.std(axis=1, ddof=0).iloc[:cutoff]
        sem_pc  = std_pc / np.sqrt(counts)
        sem_pc[counts < 2] = np.nan

        color = palette[grp_idx % len(palette)]
        ax.plot(
            mean_pc.index,
            mean_pc.values,
            label=f"{label} (n={len(group)})",
            color=color,
            linewidth=2
        )
        ax.fill_between(
            mean_pc.index,
            mean_pc.values - sem_pc.values,
            mean_pc.values + sem_pc.values,
            alpha=0.3,
            color=color
        )

    ax.set_xlabel("Time Passed (hours)", fontsize=24)
    ax.set_ylabel("Percent Correct",    fontsize=24)
    ax.set_ylim(30, 100)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.grid(True, linestyle='--', alpha=0.4)

    title = "Cumulative Accuracy"
    if len(group_labels) == 1:
        title += f" for Cohort {group_labels[0]}"
    else:
        title += " for " + " vs. ".join(group_labels)
    ax.set_title(title, fontsize=32)

    ax.legend(fontsize=16, loc='upper left')

    plt.tight_layout()
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()


def cumulative_pellets_meals(data: pd.DataFrame, bhv: int, num: int) -> None:
    """Plot cumulative pellet counts for the notebook's per-mouse inspection."""
    plt.figure(figsize=(15, 6), dpi=90)
    sns.lineplot(data=data, x='Time_passed', y='Cum_Sum', label='M1')
    plt.grid()
    plt.title(f'Cumulative Sum of Pellet for Control Group {bhv} Mice {num}', fontsize=22)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Cumulative Percentage (%)', fontsize=16)
    legend = plt.legend(title='Mice', fontsize=10)
    legend.get_title().set_fontsize(12)
    plt.show()


def calculate_accuracy(group: pd.DataFrame) -> float:
    """Return poke accuracy for the rows in ``group``.

    Args:
        group (pd.DataFrame): Table containing ``Event`` and ``Active_Poke``.

    Returns:
        float: Share of events that match the active poke.
    """
    if 'Pellet' in group['Event'].values:
        group = group[group['Event'] != 'Pellet']
        
    total_events = len(group)
    matching_events = group[group['Event'] == group['Active_Poke']]
    matching_count = len(matching_events)
    
    if total_events == 0:
        return 0
    else:
        return (matching_count / total_events) * 100


def find_inactive_index(hourly_labels: list, rev: bool):
    """Identify spans of inactive (night) or active periods within hourly labels."""
    intervals = []
    in_interval = False
    interval_start_index = None

    for i, time_str in enumerate(hourly_labels):
        current_time = datetime.strptime(time_str, '%H:%M')

        # Determine if the current time is within the inactive interval (7pm to 7am)
        if current_time.hour >= 19 or current_time.hour < 7:
            if not in_interval:
                # We are starting a new interval
                in_interval = True
                interval_start_index = i
        else:
            if in_interval:
                # We are ending the current interval
                in_interval = False
                intervals.append([interval_start_index, i - 1])
    
    # If the last time in the list is within the interval, close it
    if in_interval:
        intervals.append([interval_start_index, len(hourly_labels) - 1])
    
    hourly_labels = [datetime.strptime(time_str, '%H:%M').hour for time_str in hourly_labels]
    
    if rev:
        result = []
        start = 0
        for interval in intervals:
            if interval[0] > start:
                result.append([start, interval[0] - 1])
            start = interval[1] + 1
        if start <= len(hourly_labels) - 1:
            result.append([start, len(hourly_labels) - 1])
        return result
    return intervals


def find_learning_milestone(data: pd.DataFrame, window_hours: int = 2, accuracy_threshold: float = 0.8) -> float:
    """Return the earliest time a sliding window exceeds the accuracy threshold.

    Args:
        data (pd.DataFrame): Poke events with ``Event`` and ``Active_Poke``.
        window_hours (int): Width of the sliding window in hours.
        accuracy_threshold (float): Minimum accuracy required to declare
            learning.

    Returns:
        float: Time in hours from session start when the threshold is reached.
    """
    data = data[data['Event'] != 'Pellet'].reset_index(drop=True)
    data['is_match'] = (data['Event'] == data['Active_Poke']).astype(int)
    data['cumulative_total'] = data['is_match'].expanding().count()
    data['cumulative_match'] = data['is_match'].cumsum()
    window_timedelta = timedelta(hours=window_hours)

    for start_idx in range(len(data)):
        start_time = data.loc[start_idx, 'Time_passed']
        end_time = start_time + window_timedelta

        window_data = data[(data['Time_passed'] >= start_time) &
                           (data['Time_passed'] < end_time) &
                           (data['Event'] != 'Pellet')]
        total_events = len(window_data)
        matching_events = window_data['is_match'].sum()
        
        if total_events > 0:
            accuracy = matching_events / total_events
            if accuracy > accuracy_threshold:
                return start_time.total_seconds() / 3600  # Return the precise time that meets the condition
    # not reach high accuracy -> return session time
    return data.loc[len(data)-1, 'Time_passed'].total_seconds() / 3600