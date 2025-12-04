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
):
    """Detect meals in a session and classify them using the trained model.

    Returns a tuple of (meals_with_acc, good_mask, first_good_time) where
    * meals_with_acc is a list of [start_time, padded_accuracy_sequence]
    * good_mask is a boolean numpy array indicating model-predicted good meals
    * first_good_time is the timestamp of the first predicted good meal (or None).
    """

    meals, _ = find_meals_paper(
        data,
        time_threshold=time_threshold,
        pellet_threshold=pellet_threshold,
        in_meal_ratio=False,
        accuracy_threshold=accuracy_threshold,
        method=method,
    )

    meals_with_acc: list[list] = []
    meal_lengths: list[int] = []

    for start_time, end_time in meals:
        # Extract events for this meal using the start/end timestamps
        meal_events = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]
        
        if not meal_events.empty:
            accuracies = extract_meal_acc_each(meal_events)
            meal_lengths.append(len(accuracies))
            meals_with_acc.append([start_time, pad_meal(accuracies)])

    if not meals_with_acc:
        return [], np.zeros(0, dtype=bool), None

    candidate_idx = [idx for idx, length in enumerate(meal_lengths) if length in (2, 3, 4)]
    good_mask = np.zeros(len(meals_with_acc), dtype=bool)
    first_good_idx = None

    if candidate_idx:
        sequences = np.stack([meals_with_acc[idx][1] for idx in candidate_idx])
        predictions = predict_meal_quality(sequences, model_type=model_type)
        for pos, meal_idx in enumerate(candidate_idx):
            is_good = predictions[pos] == 0
            good_mask[meal_idx] = is_good
            if is_good and first_good_idx is None:
                first_good_idx = meal_idx
    else:
        predictions = np.array([])

    if first_good_idx is None and not candidate_idx:
        # Fallback to accuracy heuristic when model cannot evaluate the meal
        for meal_idx, (start_time, padded_acc) in enumerate(meals_with_acc):
            valid_values = [val for val in padded_acc if val != -1]
            if valid_values and np.mean(valid_values) >= accuracy_threshold:
                good_mask[meal_idx] = True
                first_good_idx = meal_idx
                break

    first_good_time = (
        pd.to_datetime(meals_with_acc[first_good_idx][0])
        if first_good_idx is not None else None
    )

    return meals_with_acc, good_mask, first_good_time


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
        return _find_meals_ipi(data, 60, 1, in_meal_ratio, 0.0)
    elif method == 'paper':
        return _find_meals_paper_original(data, time_threshold, pellet_threshold, in_meal_ratio, accuracy_threshold)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'paper' or 'ipi'.")


def _find_meals_paper_original(data:pd.DataFrame, time_threshold=60, pellet_threshold=2, in_meal_ratio=False, accuracy_threshold=50.0):
    """Original paper implementation: checks if pellet is within threshold of meal start."""
    df = data[data['Event'] == 'Pellet'].copy()
    # df['retrieval_timestamp'] = df['Time'] + pd.to_timedelta(df['collect_time'], unit='m')

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


def _find_meals_ipi(data:pd.DataFrame, time_threshold=60, pellet_threshold=2, in_meal_ratio=False, accuracy_threshold=50.0):
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
        if accuracy <= accuracy_threshold:
            continue

        meals.append([start_time, end_time])
        meal_acc.append(accuracy)
        pellets_in_meals += pellet_count
        df.loc[meal_group.index, 'meal_label'] = next_label
        next_label += 1
    
    if in_meal_ratio:
        meal_ratio = pellets_in_meals / total_pellets if total_pellets else 0.0
        return meals, meal_acc, meal_ratio
    return meals, meal_acc


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
        pellet_threshold=3,
        in_meal_ratio=True,
        method=method,
        accuracy_threshold=accuracy_threshold,
    )
    meals_with_acc, good_mask, first_meal_time = analyze_meals(data, 60, 2, 'cnn', method=method)
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