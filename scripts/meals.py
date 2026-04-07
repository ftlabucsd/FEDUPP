"""Meal detection, quality assessment, and visualisation utilities for FED3 sessions."""
from pathlib import Path
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np
from scripts.accuracy import (
    find_inactive_index,
    calculate_accuracy,
)
from scripts.meal_classifiers import predict, RNNClassifier, CNNClassifier
import torch
from scripts.preprocessing import SessionData
_MEAL_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_MEAL_MODEL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                                  'mps' if torch.backends.mps.is_available() else
                                  'cpu')
print(f"Using device: {_MEAL_MODEL_DEVICE}")
_BASE_DIR = Path(__file__).resolve().parent.parent
_MEAL_CHECKPOINTS = {
    'cnn': _BASE_DIR / 'data' / 'CNN_from_CASK.pth',
    'lstm': _BASE_DIR / 'data' / 'LSTM_from_CASK.pth',
}
plt.rcParams['figure.figsize'] = (20, 6)


def _build_meal_model(model_type: str) -> torch.nn.Module:
    """Instantiate the requested meal classifier architecture."""
    if model_type == 'lstm':
        return RNNClassifier(input_size=1, hidden_size=400, num_layers=2, num_classes=2).to(_MEAL_MODEL_DEVICE)
    if model_type == 'cnn':
        return CNNClassifier(num_classes=2, maxlen=4).to(_MEAL_MODEL_DEVICE)
    raise ValueError("Only 'lstm' and 'cnn' meal models are supported")


def _get_meal_model(model_type: str = 'cnn'):
    """Load a cached meal classifier, initialising it on demand."""
    model_type = model_type.lower()
    if model_type not in _MEAL_MODEL_CACHE:
        model = _build_meal_model(model_type)
        ckpt_path = _MEAL_CHECKPOINTS.get(model_type)
        if ckpt_path is None:
            raise ValueError("Only 'lstm' and 'cnn' meal models are supported")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Meal model checkpoint not found: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        _MEAL_MODEL_CACHE[model_type] = model
    return _MEAL_MODEL_CACHE[model_type]


def preload_meal_models(model_types: tuple[str, ...] | list[str] | None = None) -> dict[str, torch.nn.Module]:
    """Eagerly load meal classifiers so later calls avoid disk reads."""
    if model_types is None:
        model_types = ('cnn', 'lstm')

    loaded: dict[str, torch.nn.Module] = {}
    for model_type in model_types:
        key = model_type.lower()
        try:
            loaded[key] = _get_meal_model(key)
        except FileNotFoundError as exc:
            raise RuntimeWarning(f"Skipping preload for '{model_type}': {exc}")
        except Exception as exc:  # pragma: no cover - safeguard
            raise RuntimeWarning(f"Failed to preload meal model '{model_type}': {exc!r}")
    return loaded


_PRELOADED_MEAL_MODELS = preload_meal_models()


def predict_meal_quality(batch_meals, model_type: str = 'cnn'):
    """Run the chosen meal model on a batch of padded meal accuracy vectors."""
    model = _get_meal_model(model_type)
    return predict(model, batch_meals)


def analyze_meals(
    data: pd.DataFrame,
    time_threshold: int = 60,
    pellet_threshold: int = 2,
    model_type: str = 'cnn',
    accuracy_threshold: float = 50.0,
    method: str = 'paper',
    meals: list | None = None,
    *,
    strict_model_candidates: bool = True,
):
    """Detect meals in a session and classify them using the trained model.

    Returns a tuple of (meals_with_acc, good_mask, first_good_time) where
    * meals_with_acc is a list of [start_time, padded_accuracy_sequence]
    * good_mask is a boolean numpy array indicating model-predicted good meals
    * first_good_time is the timestamp of the first predicted good meal (or None).
    """

    if meals is None:
        meals, _ = find_meals_paper(
            data,
            time_threshold=time_threshold,
            pellet_threshold=pellet_threshold,
            in_meal_ratio=False,
            accuracy_threshold=accuracy_threshold,
            method=method,
        )

    meals_with_acc, meal_lengths = _extract_meals_with_acc(data, meals)
    if not meals_with_acc:
        return [], np.zeros(0, dtype=bool), None

    good_mask, first_good_time = _classify_meals(
        meals_with_acc=meals_with_acc,
        meal_lengths=meal_lengths,
        model_type=model_type,
        accuracy_threshold=accuracy_threshold,
        strict_model_candidates=strict_model_candidates,
    )

    return meals_with_acc, good_mask, first_good_time


def _extract_meals_with_acc(
    data: pd.DataFrame,
    meals: list,
) -> tuple[list[list], list[int]]:
    """Convert meal windows into padded per-meal accuracy sequences."""
    meals_with_acc: list[list] = []
    meal_lengths: list[int] = []

    for start_time, end_time in meals:
        meal_events = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]
        if meal_events.empty:
            continue
        accuracies = extract_meal_acc_each(meal_events)
        meal_lengths.append(len(accuracies))
        meals_with_acc.append([start_time, pad_meal(accuracies)])

    return meals_with_acc, meal_lengths


def _classify_meals(
    *,
    meals_with_acc: list[list],
    meal_lengths: list[int],
    model_type: str,
    accuracy_threshold: float,
    strict_model_candidates: bool,
) -> tuple[np.ndarray, pd.Timestamp | None]:
    """Classify meals as good/bad and return the first good meal time.

    Notes
    -----
    The CNN/LSTM model is only applied to meals whose accuracy-sequence length is
    2–4 (i.e., 3–5 pellets). When ``strict_model_candidates=True`` (default),
    the first-good-meal time only considers those candidate meals; non-candidate
    meals are ignored unless *no* candidate meals exist, in which case we fall
    back to the mean-accuracy heuristic.
    """
    good_mask = np.zeros(len(meals_with_acc), dtype=bool)
    first_good_idx: int | None = None

    candidate_idx = [idx for idx, length in enumerate(meal_lengths) if length in (2, 3, 4)]

    if candidate_idx:
        sequences = np.stack([meals_with_acc[idx][1] for idx in candidate_idx])
        predictions = predict_meal_quality(sequences, model_type=model_type)
        for pos, meal_idx in enumerate(candidate_idx):
            is_good = predictions[pos] == 0
            good_mask[meal_idx] = is_good
            if is_good and first_good_idx is None:
                first_good_idx = meal_idx

        if not strict_model_candidates and first_good_idx is None:
            # Allow non-candidate meals to count as good via heuristic fallback.
            for meal_idx, (_, padded_acc) in enumerate(meals_with_acc):
                if meal_idx in candidate_idx:
                    continue
                valid_values = [val for val in padded_acc if val != -1]
                if valid_values and np.mean(valid_values) >= accuracy_threshold:
                    good_mask[meal_idx] = True
                    first_good_idx = meal_idx
                    break

    if first_good_idx is None and not candidate_idx:
        # No model candidates exist -> heuristic fallback across all meals.
        for meal_idx, (_, padded_acc) in enumerate(meals_with_acc):
            valid_values = [val for val in padded_acc if val != -1]
            if valid_values and np.mean(valid_values) >= accuracy_threshold:
                good_mask[meal_idx] = True
                first_good_idx = meal_idx
                break

    first_good_time = pd.to_datetime(meals_with_acc[first_good_idx][0]) if first_good_idx is not None else None
    return good_mask, first_good_time


def pad_meal(each:list):
    """Pad a list with ``-1`` values until it reaches a length of four."""
    size = len(each)
    while size < 4:
        each.append(-1)
        size += 1
    return each


def pellet_flip(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pellet events into 10-minute bins and count occurrences."""
    data = data.set_index('Time')
    grouped_data = data[data['Event'] == 'Pellet'].resample('10min').size().reset_index()
    grouped_data.columns = ['Interval_Start', 'Pellet_Count']
    
    return grouped_data


def average_pellet(data: pd.DataFrame) -> float:
    """Calculate the average number of pellets consumed per 24 hours."""
    total_pellets = len(data[data['Event'] == 'Pellet'])
    duration = experiment_duration(data)
    if duration == 0:
        return 0.0
    return round(total_pellets / duration, 2)


def graph_pellet_frequency(grouped_data: pd.DataFrame, bhv, num, export_path=None, show: bool = False):
    """Plot 10-minute pellet counts and shade inactive periods."""
    fig, ax = plt.subplots()
    sns.barplot(data=grouped_data, x='Interval_Start', y='Pellet_Count', color='#000099', alpha=0.5, ax=ax)

    xtick_positions = ax.get_xticks()
    hourly_labels = [label.strftime('%H:%M') for label in grouped_data['Interval_Start'] if label.minute == 0]
    hourly_positions = [pos for pos, label in zip(xtick_positions, grouped_data['Interval_Start']) if label.minute == 0]

    ax.set_xticks(hourly_positions)
    ax.set_xticklabels(hourly_labels, rotation=45, horizontalalignment='right')

    if bhv is not None:
        inactive = find_inactive_index(hourly_labels, rev=False)
    else:
        inactive = find_inactive_index(hourly_labels, rev=True)

    for idx, each in enumerate(inactive):
        label = 'Inactive' if idx == 0 else None
        ax.axvspan(6*each[0], 6*(1+each[1]), color='grey', alpha=0.4, label=label)

    ax.axhline(y=5, color='red', linestyle='--', label='meal')
    if bhv is None:
        ax.set_title(f'Pellet Frequency of Mouse {num}', fontsize=18)
    else:
        ax.set_title(f'Pellet Frequency of Group {bhv} Mouse {num}', fontsize=18)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Number of Pellet', fontsize=14)
    ax.set_yticks(range(0, 19, 2))
    ax.legend()
    fig.tight_layout()
    if export_path:
        fig.savefig(export_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def find_first_accurate_meal(data:pd.DataFrame, time_threshold, pellet_threshold, model_type='cnn', method='paper'):
    """Identify the first model-classified good meal within a session."""
    meals_with_acc, good_mask, first_good_time = analyze_meals(
        data,
        time_threshold=time_threshold,
        pellet_threshold=pellet_threshold,
        model_type=model_type,
        method=method,
    )
    return meals_with_acc, first_good_time


def extract_meal_acc_each(events: pd.DataFrame):
    """Calculate per-interval pellet accuracy within a meal."""
    acc = []
    pellet_indices = events.index[events['Event'] == 'Pellet'].tolist()

    for idx in range(len(pellet_indices) - 1):
        start, end = pellet_indices[idx], pellet_indices[idx+1]
        curr_slice = events.loc[start:end]
        # curr_slice = curr_slice[curr_slice['Event'] != 'Pellet']
        acc.append(calculate_accuracy(curr_slice))

    # print(f"There are {len(pellet_indices)} pellets and {len(acc)} accuracy")
    return acc


def find_meals_paper(data:pd.DataFrame, time_threshold=60, pellet_threshold=2, in_meal_ratio=False, accuracy_threshold=50.0, method='paper'):
    """Identify meals using either the paper heuristic or interpellet interval method.
    
    Parameters
    ----------
    data : pd.DataFrame
        Session data containing pellet events
    time_threshold : int
        Time threshold in seconds for meal grouping
    pellet_threshold : int
        Minimum number of pellets required for a valid meal
    in_meal_ratio : bool
        If True, return the ratio of pellets within meals to total pellets
    accuracy_threshold : float
        Minimum accuracy percentage for a meal to be accepted
    method : str
        Meal detection method: 'paper' (default) or 'ipi' (interpellet interval)
        - 'paper': Fixed duration method - checks if meal duration is within threshold
        - 'ipi': New method - groups based on interpellet intervals between consecutive pellets
    
    Returns
    -------
    tuple
        (meals, meal_acc) or
        (meals, meal_acc, meal_ratio) if
        ``in_meal_ratio`` is True. ``meals`` is a list of [start_time, end_time]
        pairs.
    """
    if method == 'ipi':
        return _find_meals_ipi(data, time_threshold, pellet_threshold, in_meal_ratio)
    elif method == 'paper':
        return _find_meals_paper_original(data, time_threshold, pellet_threshold, in_meal_ratio, accuracy_threshold)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'paper' or 'ipi'.")


def _find_meals_paper_original(data:pd.DataFrame, time_threshold=60, pellet_threshold=2, in_meal_ratio=False, accuracy_threshold=50.0):
    """Original paper implementation: checks if pellet is within threshold of meal start."""
    df = data[data['Event'] == 'Pellet'].copy()

    total_pellets = len(df)            # denominator for the optional ratio
    pellets_in_meals = 0               # numerator we will accumulate

    meals, meal_acc = [], []

    meal_start_time = meal_end_time = None
    meal_start_idx = None
    pellet_cnt = 0

    for idx, row in df.iterrows():
        current_time = row['Time']

        # First pellet (or first after closing a meal) → open new meal window
        if meal_start_time is None:
            meal_start_time = meal_end_time = current_time
            meal_start_idx = idx
            pellet_cnt = 1
            continue

        # Same meal if retrieved within threshold of meal start
        if (current_time - meal_start_time).total_seconds() <= time_threshold:
            meal_end_time = current_time
            pellet_cnt += 1
        else:
            # Close previous burst and decide whether it's an accepted meal
            burst_events = data.loc[meal_start_idx : idx - 1]  # inclusive slice
            if pellet_cnt >= pellet_threshold:
                acc_value = calculate_accuracy(burst_events)
                if acc_value > accuracy_threshold:
                    meals.append([meal_start_time, meal_end_time])
                    meal_acc.append(acc_value)
                    pellets_in_meals += pellet_cnt

            # Start a new burst
            meal_start_time = meal_end_time = current_time
            meal_start_idx = idx
            pellet_cnt = 1

    burst_events = data.loc[meal_start_idx:]
    if pellet_cnt >= pellet_threshold:
        acc_value = calculate_accuracy(burst_events)
        if acc_value > accuracy_threshold:
            meals.append([meal_start_time, meal_end_time])
            meal_acc.append(acc_value)
            pellets_in_meals += pellet_cnt

    if in_meal_ratio:
        meal_ratio = pellets_in_meals / total_pellets if total_pellets else 0
        return meals, meal_acc, meal_ratio
    return meals, meal_acc


def _find_meals_ipi(data:pd.DataFrame, time_threshold=60, pellet_threshold=2, in_meal_ratio=False):
    """New interpellet interval (IPI) implementation: groups based on time between consecutive pellets.
    
    This method assigns pellets to meals based on the interpellet interval (IPI).
    A new meal starts whenever the time between consecutive pellets meets or exceeds the threshold.
    """
    df = data[data['Event'] == 'Pellet'].copy()
    if df.empty:
        return ([], [], 0.0) if in_meal_ratio else ([], [])
    
    # df['retrieval_timestamp'] = df['Time'] + pd.to_timedelta(df['collect_time'], unit='m')
    # df = df.sort_values('retrieval_timestamp').reset_index(drop=True)
    df = df.sort_values('Time').reset_index(drop=True)
    
    total_pellets = len(df)
    pellets_in_meals = 0

    # Calculate interpellet intervals (time since last pellet in minutes)
    df['ipi_minutes'] = df['Time'].diff().dt.total_seconds() / 60
    
    # Convert threshold from seconds to minutes for comparison
    intermeal_interval = time_threshold / 60
    
    # Identify the start of a new run: first pellet or gaps >= threshold
    new_run_mask = df['ipi_minutes'].isna() | (df['ipi_minutes'] >= intermeal_interval)
    df['run_id'] = new_run_mask.cumsum()
    df['meal_label'] = np.nan
    
    # Group by run_id and process each candidate meal
    meals, meal_acc = [], []
    next_label = 1
    
    for _, meal_group in df.groupby('run_id', sort=True):
        pellet_count = len(meal_group)
        
        # Check if meal meets pellet threshold
        if pellet_count < pellet_threshold:
            continue
        
        # Get time boundaries
        start_time = meal_group.iloc[0]['Time']
        end_time = meal_group.iloc[-1]['Time']

        # Get original data indices for this meal to calculate accuracy
        first_pellet_time = meal_group.iloc[0]['Time']
        last_pellet_time = meal_group.iloc[-1]['Time']

        # Get all events between first and last pellet of this meal
        meal_events = data[
            (data['Time'] >= first_pellet_time) & 
            (data['Time'] <= last_pellet_time)
        ]

        # Check accuracy threshold
        accuracy = calculate_accuracy(meal_events)
        meals.append([start_time, end_time])
        meal_acc.append(accuracy)
        pellets_in_meals += pellet_count
        df.loc[meal_group.index, 'meal_label'] = next_label
        next_label += 1
    
    if in_meal_ratio:
        meal_ratio = pellets_in_meals / total_pellets if total_pellets else 0.0
        return meals, meal_acc, meal_ratio
    return meals, meal_acc


def calculate_low_accuracy_meal_ratio(
    data: pd.DataFrame,
    time_threshold: int = 60,
    pellet_threshold: int = 2,
    accuracy_cutoff: float = 50.0,
    method: str = 'paper',
) -> dict:
    """Calculate the ratio of meals with accuracy below a cutoff.
    
    This finds ALL meals constrained only by time/pellet thresholds (no accuracy
    filter), then counts how many have accuracy < accuracy_cutoff.
    
    Parameters
    ----------
    data : pd.DataFrame
        Session data containing pellet events.
    time_threshold : int
        Time threshold in seconds for meal grouping.
    pellet_threshold : int
        Minimum number of pellets required for a valid meal.
    accuracy_cutoff : float
        The accuracy threshold to compare against (default 50%).
    method : str
        Meal detection method ('paper' or 'ipi').
    
    Returns
    -------
    dict
        Contains:
        - 'total_meals': Total number of meals (time/pellet constrained only)
        - 'low_accuracy_meals': Number of meals with accuracy < cutoff
        - 'high_accuracy_meals': Number of meals with accuracy >= cutoff
        - 'low_accuracy_ratio': Ratio of low-accuracy meals to total
        - 'meal_accuracies': List of all meal accuracies
    """
    # Find all meals with no accuracy threshold
    meals, meal_acc = find_meals_paper(
        data,
        time_threshold=time_threshold,
        pellet_threshold=pellet_threshold,
        in_meal_ratio=False,
        accuracy_threshold=-1.0,  # No accuracy filtering
        method=method,
    )
    
    total_meals = len(meals)
    if total_meals == 0:
        return {
            'total_meals': 0,
            'low_accuracy_meals': 0,
            'high_accuracy_meals': 0,
            'low_accuracy_ratio': 0.0,
            'meal_accuracies': [],
        }
    
    low_accuracy_count = sum(1 for acc in meal_acc if acc < accuracy_cutoff)
    high_accuracy_count = total_meals - low_accuracy_count
    
    return {
        'total_meals': total_meals,
        'low_accuracy_meals': low_accuracy_count,
        'high_accuracy_meals': high_accuracy_count,
        'low_accuracy_ratio': low_accuracy_count / total_meals,
        'meal_accuracies': meal_acc,
    }


def graphing_cum_count(data: pd.DataFrame, meal: list, bhv, num, flip=False, export_path=None, show: bool = False):
    """Plot pellet counts alongside cumulative sum and highlight meals/inactivity."""
    fig, ax1 = plt.subplots()
    ax1.plot(data['Time'], data['Pellet_Count'], color='blue')
    if bhv is None:
        ax1.set_title(f'Pellet Count and Cumulative Sum Over Time of Mouse {num}', fontsize=18)
    else:
        ax1.set_title(f'Pellet Count and Cumulative Sum Over Time of Group {bhv} Mouse {num}', fontsize=18)

    for interval in meal:
        ax1.axvspan(interval[0], interval[1], color='lightblue')

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Pellet_Count', fontsize=12)

    if not flip:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Cum_Sum', fontsize=12)
        ax2.plot(data['Time'], data['Cum_Sum'], color='blue')

    start = None
    end = None

    if bhv is not None:
        for interval in pd.date_range(start=data['Time'].min(), end=data['Time'].max(), freq='20min'):
            if (19 <= interval.hour or interval.hour < 7) and start is None:
                start = interval
            elif interval.hour == 7 and start is not None:
                ax1.axvspan(start, interval, color='grey', alpha=0.4)
                start = end = None
        if start is not None and end is None:
            ax1.axvspan(start, data['Time'].max(), color='grey', alpha=0.4)
    else:
        for interval in pd.date_range(start=data['Time'].min(), end=data['Time'].max(), freq='20min'):
            if (7 <= interval.hour < 19) and start is None:
                start = interval
            elif interval.hour == 19 and start is not None:
                ax1.axvspan(start, interval, color='grey', alpha=0.4)
                start = end = None
        if start is not None and end is None:
            ax1.axvspan(start, data['Time'].max(), color='grey', alpha=0.4)

    patch_meal = mpatches.Patch(color='lightblue', alpha=0.9, label='Meal')
    patch_inactive = mpatches.Patch(color='grey', alpha=0.5, label='Inactive')

    ax1.legend(handles=[patch_meal, patch_inactive], loc='upper right')
    if export_path:
        fig.savefig(export_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def experiment_duration(data: pd.DataFrame):
    """Return the total duration of the session in days."""
    times = pd.to_datetime(data['Time'])
    if times.empty:
        return 0.0
    duration = times.max() - times.min()
    duration_seconds = duration.total_seconds()
    return duration_seconds / (60 * 60 * 24)


def active_meal(meals: list) -> float:
    """Return the proportion of meals that begin during the active (7pm–7am) window."""
    if len(meals) == 0:
        return 0
    cnt = 0
    for meal in meals:
        if meal[0].hour >= 19 or meal[0].hour < 7:
            cnt += 1
    return round(cnt/len(meals), 4) 


def process_meal_data(session: SessionData, export_root: str | os.PathLike | None = None,
                      prefix: str | None = None, accuracy_threshold: float = 50.0,
                      method: str = 'paper'):
    """Compile per-session meal metrics and optionally export diagnostic plots."""
    data = session.raw.copy()

    meal, _, in_meal_ratio = find_meals_paper(
        data,
        time_threshold=60,
        pellet_threshold=2,
        in_meal_ratio=True,
        method=method,
        accuracy_threshold=accuracy_threshold,
    )

    # Reuse the already-detected meal windows to avoid redundant searching.
    meals_with_acc, good_mask, first_meal_time = analyze_meals(
        data,
        time_threshold=60,
        pellet_threshold=2,
        model_type='cnn',
        accuracy_threshold=accuracy_threshold,
        method=method,
        meals=meal,
    )
    meal_1 = (meal[0][0] - data['Time'][0]).total_seconds() / 3600 if meal else 0
    meal_1_good = (
        (first_meal_time - data['Time'][0]).total_seconds() / 3600
        if first_meal_time is not None
        else meal_1
    )
    group = pellet_flip(data)
    bhv, num = session.key.group, session.key.mouse_id
    
    run_prefix = prefix if prefix is not None else session.key.group.lower()
    
    session_label = session.key.session_id
    export_root_path = Path(export_root) if export_root else None
    freq_path = (
        export_root_path / f"{run_prefix}_{session_label}_pellet_frequency.svg"
        if export_root_path
        else None
    )
    cum_path = (
        export_root_path / f"{run_prefix}_{session_label}_cumulative_sum.svg"
        if export_root_path
        else None
    )
    
    graph_pellet_frequency(
        group,
        bhv,
        num,
        export_path=str(freq_path) if freq_path else None,
        show=False,
    )
    graphing_cum_count(
        data,
        meal,
        bhv,
        num,
        flip=True,
        export_path=str(cum_path) if cum_path else None,
        show=False,
    )
    
    return {
        'avg_pellet': average_pellet(data),
        'inactive_meals': active_meal(meal),
        'fir_meal': meal_1,
        'fir_good_meal': meal_1_good,
        'meal_count': round(len(meal) / experiment_duration(data), 2),
        'in_meal_ratio': in_meal_ratio,
        'good_mask': good_mask,
        'total_meals': len(good_mask),
        'meals_with_acc': meals_with_acc,
    }


def collect_good_meal_ratio(quality_map: dict) -> dict:
    """Calculate the ratio of good meals to total meals for each group.
    
    Args:
        quality_map (dict): Dictionary where keys are group names and values are lists of
            dictionaries containing 'good_mask' (boolean array) and 'total_meals' (int).
    
    Returns:
        dict: Dictionary with same keys, values are lists of good meal ratios (0.0 to 1.0).
    """
    ratios = {}
    for group, entries in quality_map.items():
        ratios[group] = []
        for entry in entries:
            good_mask = entry.get('good_mask')
            total_meals = entry.get('total_meals') or (len(good_mask) if good_mask is not None else 0)
            if good_mask is None or total_meals == 0:
                ratios[group].append(0.0)
            else:
                ratios[group].append(float(good_mask.sum()) / total_meals)
    return ratios


def find_meals_by_blocks(
    blocks: list[pd.DataFrame],
    time_threshold: int = 60,
    pellet_threshold: int = 2,
    method: str = 'paper',
    accuracy_threshold: float = 50.0,
) -> tuple[list, list, list]:
    """Detect meals within each block separately, ensuring no cross-block meals.
    
    This function processes each block independently to detect meals, then combines
    all block meals into a session-level list. This ensures meal boundaries respect
    block boundaries (active poke switches).
    
    Args:
        blocks (list[pd.DataFrame]): List of block DataFrames from split_data_to_blocks.
        time_threshold (int): Maximum seconds between pellets within a meal.
        pellet_threshold (int): Minimum number of pellets required for a valid meal.
        method (str): Meal detection method ('paper' or 'ipi').
        accuracy_threshold (float): Minimum accuracy for a meal to be accepted.
    
    Returns:
        tuple: Contains:
            - session_meals (list): Combined list of all meals from all blocks.
                Each meal is [start_time, end_time].
            - session_meal_acc (list): Accuracy values for each meal.
            - block_meal_info (list): Per-block meal information. Each entry is a dict:
                {
                    'block_idx': int,
                    'meals': list of [start_time, end_time],
                    'meal_acc': list of accuracy values,
                    'in_meal_ratio': float
                }
    """
    session_meals = []
    session_meal_acc = []
    block_meal_info = []
    
    for block_idx, block in enumerate(blocks):
        if block.empty:
            block_meal_info.append({
                'block_idx': block_idx,
                'meals': [],
                'meal_acc': [],
                'in_meal_ratio': 0.0,
            })
            continue
        
        # Detect meals within this block
        result = find_meals_paper(
            block,
            time_threshold=time_threshold,
            pellet_threshold=pellet_threshold,
            in_meal_ratio=True,
            accuracy_threshold=accuracy_threshold,
            method=method,
        )
        block_meals, block_acc, in_meal_ratio = result
        
        # Store per-block info
        block_meal_info.append({
            'block_idx': block_idx,
            'meals': block_meals,
            'meal_acc': block_acc,
            'in_meal_ratio': in_meal_ratio,
        })
        
        # Accumulate to session-level lists
        session_meals.extend(block_meals)
        session_meal_acc.extend(block_acc)
    
    return session_meals, session_meal_acc, block_meal_info


def analyze_meals_by_blocks(
    blocks: list[pd.DataFrame],
    time_threshold: int = 60,
    pellet_threshold: int = 2,
    model_type: str = 'cnn',
    accuracy_threshold: float = 50.0,
    method: str = 'paper',
) -> dict:
    """Detect and analyze meals within blocks, returning comprehensive session metrics.
    
    This function combines block-based meal detection with ML-based quality classification.
    It ensures no cross-block meals and provides both session-level and block-level metrics.
    
    Args:
        blocks (list[pd.DataFrame]): List of block DataFrames from split_data_to_blocks.
        time_threshold (int): Maximum seconds between pellets within a meal.
        pellet_threshold (int): Minimum number of pellets required for a valid meal.
        model_type (str): Model type for quality prediction ('cnn' or 'lstm').
        accuracy_threshold (float): Minimum accuracy for a meal to be accepted.
        method (str): Meal detection method ('paper' or 'ipi').
    
    Returns:
        dict: Comprehensive meal analysis results:
            - 'session_meals': Combined list of [start_time, end_time] from all blocks.
            - 'session_meal_acc': Accuracy values for each meal.
            - 'block_meal_info': Per-block detailed info (meals, accuracy, ratio).
            - 'meals_with_acc': List of [start_time, padded_accuracy_sequence].
            - 'good_mask': Boolean array of ML predictions (True = good meal).
            - 'first_good_time': Timestamp of first good meal or None.
            - 'in_meal_ratio': Overall ratio of pellets within meals.
            - 'total_meals': Total number of detected meals.
    """
    # Get block-based meals
    session_meals, session_meal_acc, block_meal_info = find_meals_by_blocks(
        blocks=blocks,
        time_threshold=time_threshold,
        pellet_threshold=pellet_threshold,
        method=method,
        accuracy_threshold=accuracy_threshold,
    )
    
    if not session_meals:
        return {
            'session_meals': [],
            'session_meal_acc': [],
            'block_meal_info': block_meal_info,
            'meals_with_acc': [],
            'good_mask': np.zeros(0, dtype=bool),
            'first_good_time': None,
            'in_meal_ratio': 0.0,
            'total_meals': 0,
        }
    
    # Combine all blocks for extracting meal accuracy sequences
    # (we need the full data to get events within each meal window)
    combined_data = pd.concat(blocks, ignore_index=True)
    
    # Extract accuracy sequences for each meal
    meals_with_acc, meal_lengths = _extract_meals_with_acc(combined_data, session_meals)
    good_mask, first_good_time = _classify_meals(
        meals_with_acc=meals_with_acc,
        meal_lengths=meal_lengths,
        model_type=model_type,
        accuracy_threshold=accuracy_threshold,
        strict_model_candidates=True,
    )
    
    # Calculate overall in-meal ratio
    total_pellets = sum(len(block[block['Event'] == 'Pellet']) for block in blocks)
    pellets_in_meals = sum(
        len(combined_data[
            (combined_data['Time'] >= start) & 
            (combined_data['Time'] <= end) &
            (combined_data['Event'] == 'Pellet')
        ])
        for start, end in session_meals
    )
    in_meal_ratio = pellets_in_meals / total_pellets if total_pellets > 0 else 0.0
    
    return {
        'session_meals': session_meals,
        'session_meal_acc': session_meal_acc,
        'block_meal_info': block_meal_info,
        'meals_with_acc': meals_with_acc,
        'good_mask': good_mask,
        'first_good_time': first_good_time,
        'in_meal_ratio': in_meal_ratio,
        'total_meals': len(session_meals),
    }


def process_meal_data_with_blocks(
    session: SessionData,
    blocks: list[pd.DataFrame] | None = None,
    meal_analysis: dict | None = None,
    export_root: str | os.PathLike | None = None,
    prefix: str | None = None,
    accuracy_threshold: float = 50.0,
    method: str = 'paper',
) -> dict:
    """Process meal data using block-based detection for consistency.
    
    For FR1 sessions, the entire session is treated as a single block.
    For Reversal sessions, pre-computed blocks should be provided.
    
    Args:
        session (SessionData): Session data object with raw data.
        blocks (list[pd.DataFrame], optional): Pre-computed blocks for reversal sessions.
            If None, the entire session is treated as one block (FR1 behavior).
        export_root (str | PathLike, optional): Directory to save diagnostic plots.
        prefix (str, optional): Prefix for exported filenames.
        accuracy_threshold (float): Minimum accuracy for meal acceptance.
        method (str): Meal detection method ('paper' or 'ipi').
    
    Returns:
        dict: Comprehensive meal metrics including:
            - 'avg_pellet': Pellets per day
            - 'inactive_meals': Proportion of meals during inactive periods
            - 'fir_meal': First meal time (hours from session start)
            - 'fir_good_meal': First good meal time (hours)
            - 'meal_count': Meals per day
            - 'in_meal_ratio': Fraction of pellets within meals
            - 'good_mask': Boolean array of ML quality predictions
            - 'total_meals': Number of detected meals
            - 'meals_with_acc': List of [start_time, padded_accuracy]
            - 'block_meal_info': Per-block meal details
    """
    data = session.raw.copy()
    
    # If no blocks provided, treat entire session as one block (FR1 case)
    if blocks is None:
        blocks = [data]
    
    # Reuse pre-computed analysis when available to avoid redundant meal detection.
    analysis = meal_analysis
    if analysis is None:
        analysis = analyze_meals_by_blocks(
            blocks=blocks,
            time_threshold=60,
            pellet_threshold=2,
            model_type='cnn',
            accuracy_threshold=accuracy_threshold,
            method=method,
        )
    
    session_meals = analysis['session_meals']
    first_meal_time = analysis['first_good_time']
    
    # Calculate timing metrics
    session_start = data['Time'].iloc[0] if not data.empty else None
    meal_1 = 0
    if session_meals and session_start is not None:
        meal_1 = (session_meals[0][0] - session_start).total_seconds() / 3600
    
    meal_1_good = meal_1
    if first_meal_time is not None and session_start is not None:
        meal_1_good = (first_meal_time - session_start).total_seconds() / 3600
    
    # Generate diagnostic plots
    bhv, num = session.key.group, session.key.mouse_id
    run_prefix = prefix if prefix is not None else session.key.group.lower()
    session_label = session.key.session_id
    
    group = pellet_flip(data)
    
    export_root_path = Path(export_root) if export_root else None
    freq_path = (
        export_root_path / f"{run_prefix}_{session_label}_pellet_frequency.svg"
        if export_root_path else None
    )
    cum_path = (
        export_root_path / f"{run_prefix}_{session_label}_cumulative_sum.svg"
        if export_root_path else None
    )
    
    graph_pellet_frequency(
        group, bhv, num,
        export_path=str(freq_path) if freq_path else None,
        show=False,
    )
    graphing_cum_count(
        data, session_meals, bhv, num,
        flip=True,
        export_path=str(cum_path) if cum_path else None,
        show=False,
    )
    
    duration = experiment_duration(data)
    
    return {
        'avg_pellet': average_pellet(data),
        'inactive_meals': active_meal(session_meals),
        'fir_meal': meal_1,
        'fir_good_meal': meal_1_good,
        'meal_count': round(len(session_meals) / duration, 2) if duration > 0 else 0,
        'in_meal_ratio': analysis['in_meal_ratio'],
        'good_mask': analysis['good_mask'],
        'total_meals': analysis['total_meals'],
        'meals_with_acc': analysis['meals_with_acc'],
        'block_meal_info': analysis['block_meal_info'],
    }


def _build_meal_records(
    data: pd.DataFrame,
    meals: list,
    meal_acc: list,
    *,
    session: SessionData,
    session_type: str,
    accuracy_cutoff: float,
) -> pd.DataFrame:
    """Build a meal-level table for one session."""
    if data.empty or not meals:
        return pd.DataFrame(
            columns=[
                'Session_Type',
                'Group',
                'Mouse_ID',
                'Session_ID',
                'Meal_Index',
                'Meal_Start',
                'Meal_End',
                'Time_From_Start_Hours',
                'Meal_Duration_Minutes',
                'Pellet_Count',
                'Accuracy',
                'Accuracy_Class',
            ]
        )

    session_start = pd.to_datetime(data['Time'].iloc[0])
    rows: list[dict[str, str | int | float | pd.Timestamp]] = []

    for meal_idx, ((meal_start, meal_end), acc_value) in enumerate(zip(meals, meal_acc), start=1):
        meal_events = data[
            (data['Time'] >= meal_start)
            & (data['Time'] <= meal_end)
        ]
        pellet_count = int((meal_events['Event'] == 'Pellet').sum())
        rows.append(
            {
                'Session_Type': session_type,
                'Group': session.key.group,
                'Mouse_ID': session.key.mouse_id,
                'Session_ID': session.key.session_id,
                'Meal_Index': meal_idx,
                'Meal_Start': pd.to_datetime(meal_start),
                'Meal_End': pd.to_datetime(meal_end),
                'Time_From_Start_Hours': (
                    (pd.to_datetime(meal_start) - session_start).total_seconds() / 3600
                ),
                'Meal_Duration_Minutes': (
                    (pd.to_datetime(meal_end) - pd.to_datetime(meal_start)).total_seconds() / 60
                ),
                'Pellet_Count': pellet_count,
                'Accuracy': float(acc_value),
                'Accuracy_Class': (
                    f'>= {accuracy_cutoff:.0f}%'
                    if float(acc_value) >= accuracy_cutoff
                    else f'< {accuracy_cutoff:.0f}%'
                ),
            }
        )

    return pd.DataFrame(rows)


def _group_accuracy_breakdown(meal_level_df: pd.DataFrame, *, accuracy_cutoff: float) -> dict[str, dict]:
    """Aggregate low/high accuracy meal counts for one session type."""
    stats_by_group: dict[str, dict] = {}
    if meal_level_df.empty:
        return stats_by_group

    for group, group_df in meal_level_df.groupby('Group', sort=False):
        total = int(len(group_df))
        low_acc = int((group_df['Accuracy'] < accuracy_cutoff).sum())
        high_acc = int(total - low_acc)
        stats_by_group[group] = {
            'total': total,
            'low_acc': low_acc,
            'high_acc': high_acc,
            'low_accuracy_ratio': (low_acc / total) if total > 0 else 0.0,
            'high_accuracy_ratio': (high_acc / total) if total > 0 else 0.0,
        }
    return stats_by_group


def _avg_meal_size_by_group(meal_level_df: pd.DataFrame) -> dict[str, list[float]]:
    """Return per-session mean pellet counts grouped by cohort."""
    if meal_level_df.empty:
        return {}

    out: dict[str, list[float]] = {}
    grouped = meal_level_df.groupby(['Group', 'Session_ID'], sort=False)
    for (group, _session_id), session_df in grouped:
        out.setdefault(group, []).append(float(session_df['Pellet_Count'].mean()))
    return out


def _meal_meta_group_summary(
    meal_level_df: pd.DataFrame,
    session_level_df: pd.DataFrame,
    *,
    accuracy_cutoff: float,
) -> pd.DataFrame:
    """Create a compact group-level summary table for meal metadata."""
    columns = [
        'Session_Type',
        'Group',
        'n_sessions',
        'n_meals',
        'mean_meals_per_session',
        'mean_pellets_per_meal',
        'median_pellets_per_meal',
        'pellet_count_iqr',
        'low_accuracy_pct',
        'high_accuracy_pct',
    ]
    if session_level_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, str | int | float]] = []
    for (session_type, group), session_group_df in session_level_df.groupby(['Session_Type', 'Group'], sort=False):
        meal_group_df = meal_level_df[
            (meal_level_df['Session_Type'] == session_type)
            & (meal_level_df['Group'] == group)
        ]
        pellet_counts = meal_group_df['Pellet_Count'].to_numpy(dtype=float) if not meal_group_df.empty else np.array([])
        accuracies = meal_group_df['Accuracy'].to_numpy(dtype=float) if not meal_group_df.empty else np.array([])
        low_pct = float(np.mean(accuracies < accuracy_cutoff) * 100) if accuracies.size > 0 else np.nan
        high_pct = float(np.mean(accuracies >= accuracy_cutoff) * 100) if accuracies.size > 0 else np.nan
        pellet_iqr = (
            float(np.percentile(pellet_counts, 75) - np.percentile(pellet_counts, 25))
            if pellet_counts.size > 0
            else np.nan
        )
        rows.append(
            {
                'Session_Type': session_type,
                'Group': group,
                'n_sessions': int(len(session_group_df)),
                'n_meals': int(len(meal_group_df)),
                'mean_meals_per_session': float(session_group_df['Total_Meals_All'].mean()),
                'mean_pellets_per_meal': float(np.mean(pellet_counts)) if pellet_counts.size > 0 else np.nan,
                'median_pellets_per_meal': float(np.median(pellet_counts)) if pellet_counts.size > 0 else np.nan,
                'pellet_count_iqr': pellet_iqr,
                'low_accuracy_pct': low_pct,
                'high_accuracy_pct': high_pct,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _session_rows_to_frame(session_rows: list[dict[str, str | int | float]]) -> pd.DataFrame:
    """Convert per-session row dicts into a standard DataFrame."""
    return pd.DataFrame(
        session_rows,
        columns=['Session_Type', 'Group', 'Mouse_ID', 'Session_ID', 'Total_Meals_All'],
    )


def _filter_meal_meta_for_accuracy_threshold(
    meal_level_df: pd.DataFrame,
    session_level_df: pd.DataFrame,
    *,
    accuracy_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter meal metadata to meals meeting the reporting threshold.

    The returned session-level frame preserves all sessions and sets the
    per-session meal count to zero when a session has no meals that pass the
    threshold.
    """
    if meal_level_df.empty:
        return meal_level_df.copy(), session_level_df.copy()

    filtered_meal_level_df = meal_level_df[
        meal_level_df['Accuracy'] >= accuracy_cutoff
    ].copy()

    if session_level_df.empty:
        return filtered_meal_level_df, session_level_df.copy()

    key_cols = ['Session_Type', 'Group', 'Mouse_ID', 'Session_ID']
    filtered_counts = (
        filtered_meal_level_df
        .groupby(key_cols, sort=False)
        .size()
        .reset_index(name='Total_Meals_All')
    )
    filtered_session_level_df = session_level_df[key_cols].merge(
        filtered_counts,
        on=key_cols,
        how='left',
    )
    filtered_session_level_df['Total_Meals_All'] = (
        filtered_session_level_df['Total_Meals_All']
        .fillna(0)
        .astype(int)
    )
    return filtered_meal_level_df, filtered_session_level_df


def build_low_accuracy_summary_table(
    accuracy_stats_by_type: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    """Create a tidy low/high-accuracy summary table for display."""
    rows: list[dict[str, str | int | float]] = []
    for session_type, stats_by_group in accuracy_stats_by_type.items():
        total_all = 0
        low_all = 0
        high_all = 0
        for group, stats in stats_by_group.items():
            total = int(stats.get('total', 0))
            low_acc = int(stats.get('low_acc', 0))
            high_acc = int(stats.get('high_acc', 0))
            total_all += total
            low_all += low_acc
            high_all += high_acc
            rows.append(
                {
                    'Session_Type': session_type,
                    'Group': group,
                    'Total_Meals': total,
                    'Low_Accuracy_Meals': low_acc,
                    'High_Accuracy_Meals': high_acc,
                    'Low_Accuracy_Pct': float(stats.get('low_accuracy_ratio', 0.0) * 100),
                    'High_Accuracy_Pct': float(stats.get('high_accuracy_ratio', 0.0) * 100),
                }
            )

        if total_all > 0:
            rows.append(
                {
                    'Session_Type': session_type,
                    'Group': 'Global',
                    'Total_Meals': total_all,
                    'Low_Accuracy_Meals': low_all,
                    'High_Accuracy_Meals': high_all,
                    'Low_Accuracy_Pct': (low_all / total_all) * 100,
                    'High_Accuracy_Pct': (high_all / total_all) * 100,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            'Session_Type',
            'Group',
            'Total_Meals',
            'Low_Accuracy_Meals',
            'High_Accuracy_Meals',
            'Low_Accuracy_Pct',
            'High_Accuracy_Pct',
        ],
    )


def build_avg_meal_size_summary_table(
    avg_meal_size_by_type: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    """Create a tidy table of per-group average meal-size summaries."""
    rows: list[dict[str, str | int | float]] = []
    for session_type, avg_size_map in avg_meal_size_by_type.items():
        for group, values in avg_size_map.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            rows.append(
                {
                    'Session_Type': session_type,
                    'Group': group,
                    'n_sessions': int(arr.size),
                    'Mean_Pellets_Per_Meal': float(np.mean(arr)) if arr.size > 0 else np.nan,
                    'SD_Pellets_Per_Meal': float(np.std(arr)) if arr.size > 0 else np.nan,
                    'Median_Pellets_Per_Meal': float(np.median(arr)) if arr.size > 0 else np.nan,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            'Session_Type',
            'Group',
            'n_sessions',
            'Mean_Pellets_Per_Meal',
            'SD_Pellets_Per_Meal',
            'Median_Pellets_Per_Meal',
        ],
    )


def build_meal_histogram_bin_table(meal_level_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize meal counts into wide histogram bins by session type and group.

    Returns one row per ``Session_Type`` / ``Group`` and one column per pellet
    count bin, e.g. ``2_pellet_meals``.
    """
    key_cols = ['Session_Type', 'Group']
    if meal_level_df.empty:
        return pd.DataFrame(columns=key_cols)

    working_df = meal_level_df[key_cols + ['Pellet_Count']].copy()
    working_df['Pellet_Count'] = pd.to_numeric(working_df['Pellet_Count'], errors='coerce')
    working_df = working_df.dropna(subset=['Pellet_Count'])
    if working_df.empty:
        return pd.DataFrame(columns=key_cols)

    working_df['Pellet_Count'] = working_df['Pellet_Count'].astype(int)
    counts_df = (
        working_df
        .groupby(key_cols + ['Pellet_Count'], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    pellet_cols = [col for col in counts_df.columns if isinstance(col, (int, np.integer))]
    rename_map = {
        pellet_count: f"{int(pellet_count)}_pellet_meals"
        for pellet_count in pellet_cols
    }
    counts_df = counts_df.rename(columns=rename_map)

    ordered_bin_cols = [rename_map[pellet_count] for pellet_count in sorted(pellet_cols)]
    return counts_df[key_cols + ordered_bin_cols]


def collect_meal_meta(
    *,
    session_type: str,
    group_sessions: dict[str, list[SessionData]] | None = None,
    session_analyses: dict[str, list[dict]] | None = None,
    export_root: str | os.PathLike | None = None,
    prefix_map: dict[str, str] | None = None,
    time_threshold: int = 60,
    pellet_threshold: int = 2,
    accuracy_cutoff: float = 50.0,
    method: str = 'paper',
) -> dict:
    """Collect meal metadata for one paradigm without recomputing later in the notebook."""
    session_type = str(session_type).upper()
    if session_type not in {'FR1', 'REV'}:
        raise ValueError("session_type must be 'FR1' or 'REV'.")

    metrics_by_group: dict[str, list[dict]] = {}
    meal_level_frames: list[pd.DataFrame] = []
    session_rows: list[dict[str, str | int | float]] = []

    if session_type == 'FR1':
        if group_sessions is None:
            raise ValueError("group_sessions is required for FR1 meal metadata.")
        metrics_by_group = {group: [] for group in group_sessions}
        for group, sessions in group_sessions.items():
            run_prefix = prefix_map.get(group, group.lower()) if prefix_map else group.lower()
            for session in sessions:
                metrics = process_meal_data_with_blocks(
                    session,
                    blocks=[session.raw.copy()],
                    export_root=export_root,
                    prefix=run_prefix,
                    accuracy_threshold=accuracy_cutoff,
                    method=method,
                )
                metrics_by_group[group].append(metrics)

                all_meals, all_meal_acc, _ = find_meals_by_blocks(
                    [session.raw.copy()],
                    time_threshold=time_threshold,
                    pellet_threshold=pellet_threshold,
                    method=method,
                    accuracy_threshold=-1.0,
                )
                meal_level_frames.append(
                    _build_meal_records(
                        session.raw.copy(),
                        all_meals,
                        all_meal_acc,
                        session=session,
                        session_type=session_type,
                        accuracy_cutoff=accuracy_cutoff,
                    )
                )
                session_rows.append(
                    {
                        'Session_Type': session_type,
                        'Group': group,
                        'Mouse_ID': session.key.mouse_id,
                        'Session_ID': session.key.session_id,
                        'Total_Meals_All': int(len(all_meals)),
                    }
                )
    else:
        if session_analyses is None:
            raise ValueError("session_analyses is required for REV meal metadata.")
        metrics_by_group = {group: [] for group in session_analyses}
        for group, analyses in session_analyses.items():
            run_prefix = prefix_map.get(group, group.lower()) if prefix_map else group.lower()
            for analysis in analyses:
                session = analysis['session']
                blocks = analysis.get('blocks', [])
                if not blocks:
                    continue

                metrics = process_meal_data_with_blocks(
                    session,
                    blocks=blocks,
                    meal_analysis=analysis.get('meal_analysis'),
                    export_root=export_root,
                    prefix=run_prefix,
                    accuracy_threshold=accuracy_cutoff,
                    method=method,
                )
                metrics_by_group[group].append(metrics)

                limited_data = pd.concat(blocks, ignore_index=True)
                all_meals, all_meal_acc, _ = find_meals_by_blocks(
                    blocks,
                    time_threshold=time_threshold,
                    pellet_threshold=pellet_threshold,
                    method=method,
                    accuracy_threshold=-1.0,
                )
                meal_level_frames.append(
                    _build_meal_records(
                        limited_data,
                        all_meals,
                        all_meal_acc,
                        session=session,
                        session_type=session_type,
                        accuracy_cutoff=accuracy_cutoff,
                    )
                )
                session_rows.append(
                    {
                        'Session_Type': session_type,
                        'Group': group,
                        'Mouse_ID': session.key.mouse_id,
                        'Session_ID': session.key.session_id,
                        'Total_Meals_All': int(len(all_meals)),
                    }
                )

    meal_level_df = pd.concat(meal_level_frames, ignore_index=True) if meal_level_frames else pd.DataFrame()
    session_level_df = _session_rows_to_frame(session_rows)
    quality_map = {
        group: [
            {
                'good_mask': metrics.get('good_mask'),
                'total_meals': metrics.get('total_meals'),
            }
            for metrics in metrics_list
        ]
        for group, metrics_list in metrics_by_group.items()
    }
    accuracy_stats = _group_accuracy_breakdown(meal_level_df, accuracy_cutoff=accuracy_cutoff)
    avg_meal_size = _avg_meal_size_by_group(meal_level_df)

    return {
        'session_type': session_type,
        'metrics_by_group': metrics_by_group,
        'quality_map': quality_map,
        'good_meal_ratio': collect_good_meal_ratio(quality_map),
        'meal_level': meal_level_df,
        'session_level': session_level_df,
        'accuracy_stats': accuracy_stats,
        'avg_meal_size': avg_meal_size,
    }


def combine_meal_meta(
    meal_meta_items: list[dict],
    *,
    accuracy_cutoff: float = 50.0,
) -> dict:
    """Combine cached per-paradigm meal metadata into shared tables.

    Reviewer-facing histogram and meal-size summary tables are based on meals
    that meet the reporting cutoff (>= ``accuracy_cutoff``), while the
    low/high-accuracy breakdown remains based on all meals.
    """
    meal_level_frames = []
    session_level_frames = []
    accuracy_stats_by_type: dict[str, dict[str, dict]] = {}
    avg_meal_size_by_type: dict[str, dict[str, list[float]]] = {}

    for item in meal_meta_items:
        if not item:
            continue
        session_type = item.get('session_type')
        if session_type is None:
            continue
        meal_level_frames.append(item.get('meal_level', pd.DataFrame()))
        session_level_frames.append(item.get('session_level', pd.DataFrame()))
        accuracy_stats_by_type[session_type] = item.get('accuracy_stats', {})
        avg_meal_size_by_type[session_type] = item.get('avg_meal_size', {})

    meal_level_df = pd.concat(meal_level_frames, ignore_index=True) if meal_level_frames else pd.DataFrame()
    session_level_df = pd.concat(session_level_frames, ignore_index=True) if session_level_frames else pd.DataFrame()
    filtered_meal_level_df, filtered_session_level_df = _filter_meal_meta_for_accuracy_threshold(
        meal_level_df,
        session_level_df,
        accuracy_cutoff=accuracy_cutoff,
    )
    group_summary_df = _meal_meta_group_summary(
        filtered_meal_level_df,
        filtered_session_level_df,
        accuracy_cutoff=accuracy_cutoff,
    )
    filtered_avg_meal_size_by_type = {
        session_type: _avg_meal_size_by_group(
            filtered_meal_level_df[filtered_meal_level_df['Session_Type'] == session_type]
        )
        for session_type in accuracy_stats_by_type
    }

    return {
        'meal_level': filtered_meal_level_df,
        'session_level': filtered_session_level_df,
        'meal_level_all': meal_level_df,
        'session_level_all': session_level_df,
        'group_summary': group_summary_df,
        'low_accuracy_summary': build_low_accuracy_summary_table(accuracy_stats_by_type),
        'avg_meal_size_summary': build_avg_meal_size_summary_table(filtered_avg_meal_size_by_type),
        'accuracy_stats_by_type': accuracy_stats_by_type,
        'avg_meal_size_by_type': filtered_avg_meal_size_by_type,
    }


def build_meal_meta_summary(
    *,
    fr1_group_sessions: dict[str, list[SessionData]] | None = None,
    rev_session_analyses: dict[str, list[dict]] | None = None,
    fr1_export_root: str | os.PathLike | None = None,
    rev_export_root: str | os.PathLike | None = None,
    prefix_map: dict[str, str] | None = None,
    time_threshold: int = 60,
    pellet_threshold: int = 2,
    accuracy_cutoff: float = 50.0,
    method: str = 'paper',
) -> dict:
    """Build combined FR1/Reversal meal metadata tables and per-session metrics."""
    meal_meta_items: list[dict] = []
    session_metrics_by_type: dict[str, dict[str, list[dict]]] = {}

    if fr1_group_sessions is not None:
        fr1_meta = collect_meal_meta(
            session_type='FR1',
            group_sessions=fr1_group_sessions,
            export_root=fr1_export_root,
            prefix_map=prefix_map,
            time_threshold=time_threshold,
            pellet_threshold=pellet_threshold,
            accuracy_cutoff=accuracy_cutoff,
            method=method,
        )
        meal_meta_items.append(fr1_meta)
        session_metrics_by_type['FR1'] = fr1_meta['metrics_by_group']

    if rev_session_analyses is not None:
        rev_meta = collect_meal_meta(
            session_type='REV',
            session_analyses=rev_session_analyses,
            export_root=rev_export_root,
            prefix_map=prefix_map,
            time_threshold=time_threshold,
            pellet_threshold=pellet_threshold,
            accuracy_cutoff=accuracy_cutoff,
            method=method,
        )
        meal_meta_items.append(rev_meta)
        session_metrics_by_type['REV'] = rev_meta['metrics_by_group']

    combined = combine_meal_meta(meal_meta_items, accuracy_cutoff=accuracy_cutoff)
    combined['session_metrics_by_type'] = session_metrics_by_type
    return combined


def plot_meal_meta_summary(
    meal_level_df: pd.DataFrame,
    *,
    export_path: str | os.PathLike | None = None,
    show: bool = True,
) -> None:
    """Plot FR1/Reversal meal-length distributions as histograms only."""
    if meal_level_df.empty:
        return

    session_types = [
        session_type
        for session_type in ['FR1', 'REV']
        if session_type in meal_level_df['Session_Type'].astype(str).unique()
    ]
    if not session_types:
        return

    fig, axes = plt.subplots(
        len(session_types),
        1,
        figsize=(12, 4.8 * len(session_types)),
        sharex=False,
    )
    if len(session_types) == 1:
        axes = np.array([axes])

    for row_idx, session_type in enumerate(session_types):
        ax_hist = axes[row_idx]

        plot_df = meal_level_df[meal_level_df['Session_Type'] == session_type].copy()
        if plot_df.empty:
            ax_hist.set_visible(False)
            continue

        max_pellet_count = int(plot_df['Pellet_Count'].max())
        bin_edges = np.arange(1.5, max_pellet_count + 1.6, 1.0)
        sns.histplot(
            data=plot_df,
            x='Pellet_Count',
            hue='Group',
            bins=bin_edges,
            multiple='dodge',
            element='bars',
            fill=True,
            stat='count',
            common_norm=False,
            edgecolor='black',
            linewidth=1.0,
            alpha=0.8,
            shrink=0.95,
            ax=ax_hist,
        )
        ax_hist.set_xlabel('Meal length / pellet count')
        ax_hist.set_ylabel('Meal count')
        ax_hist.set_title(f'{session_type}: meal length distribution')
        ax_hist.set_xticks(np.arange(2, max_pellet_count + 1))
        ax_hist.grid(axis='y', alpha=0.25)
        legend = ax_hist.get_legend()
        if legend is not None:
            legend.set_title('Group')

    fig.suptitle(
        'Meal meta summary: meal length distributions',
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if export_path:
        fig.savefig(export_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)