"""
This script analyzes behavioral data from FED3 experiments, focusing on transitions
between different states (e.g., left/right pokes) and learning performance. It
includes functions to split data into blocks, calculate transition statistics,
visualize learning trends, and assess learning scores.
"""
import os
from itertools import combinations
from datetime import timedelta
from scripts.utils import graph_group_stats, palette, run_pairwise_tests

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

from scripts.meals import (
    find_meals_paper,
    find_first_accurate_meal,
    find_meals_by_blocks,
    analyze_meals_by_blocks,
)


def split_data_to_blocks(data_dropped: pd.DataFrame, day: int = 3) -> list[pd.DataFrame]:
    """Group consecutive rows into blocks that share the same active poke.

    Args:
        data_dropped (pd.DataFrame): Preprocessed session events containing
            ``Active_Poke`` and ``Time_passed`` columns.
        day (int): Only rows collected within this many days from session start
            are considered when forming blocks.

    Returns:
        list[pd.DataFrame]: Every block represents a continuous stretch of
        events where the active poke does not change.
    """
    data_dropped = data_dropped[data_dropped['Time_passed'] < timedelta(days=day)]
    curr_poke = data_dropped['Active_Poke'][0]
    blocks: list[pd.DataFrame] = []
    start_idx = 0

    for key, val in data_dropped.iterrows():
        if val['Active_Poke'] != curr_poke:
            blocks.append(data_dropped.iloc[start_idx:key].reset_index(drop=True))
            start_idx = key
            curr_poke = val['Active_Poke']

    blocks.append(data_dropped.iloc[start_idx:].reset_index(drop=True))
    return blocks


def count_transitions(sub_frame: pd.DataFrame) -> dict[str, int]:
    """Count whether poke events stay on the same side or switch sides.

    Args:
        sub_frame (pd.DataFrame): Block data containing ``Event`` and
            ``Active_Poke`` columns.

    Returns:
        dict[str, int]: Totals for each left/right transition direction plus
        the number of successful pokes (``success_count``).
    """
    transitions = {
        'Left_to_Left': 0,
        'Left_to_Right': 0,
        'Right_to_Right': 0,
        'Right_to_Left': 0,
        'success_count' : 0,
    }

    prev_event = None
    
    for _, row in sub_frame.iterrows():
        event = row['Event']

        if prev_event is not None:
            transition = f"{prev_event}_to_{event}"
            if transition in transitions:
                transitions[transition] += 1
        
        if event == row['Active_Poke']:
            transitions['success_count'] += 1

        prev_event = event

    return transitions

def count_pellet(sub_frame: pd.DataFrame) -> int:
    """Count pellet events within the provided block.

    Args:
        sub_frame (pd.DataFrame): Block data filtered to the rows of interest.

    Returns:
        int: Number of rows whose ``Event`` contains the word ``Pellet``.
    """
    pellet_count = 0
    
    for _, row in sub_frame.iterrows():
        event = row['Event']
        
        if 'Pellet' in event:
            pellet_count += 1
    
    return pellet_count


def remove_pellet(block: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``block`` without pellet rows.

    Transition metrics focus on poke actions, so pellet events are excluded
    before computing success rates.
    """
    return block[block['Event'] != 'Pellet']


def get_transition_info(
    blocks: list[pd.DataFrame],
    meal_config: list,
    reverse: bool,
    method: str = 'paper',
    block_meal_info: list | None = None,
    first_good_times: list | None = None,
) -> pd.DataFrame:
    """Calculate block-level transition, meal, and activity statistics.

    Args:
        blocks (list[pd.DataFrame]): Sequence of block DataFrames produced by
            ``split_data_to_blocks``.
        meal_config (list): ``[time_threshold, pellet_threshold]`` arguments to
            reuse when calling ``find_meals_paper``.
        reverse (bool): When True, treat traditionally inactive blocks as
            active. This keeps plot shading consistent across FR1 and REV
            sessions.
        method (str): Meal detection method ('paper' or 'ipi').
        block_meal_info (list, optional): Pre-computed per-block meal info from
            ``find_meals_by_blocks``. If provided, reuses this data instead of
            recomputing meals for each block.
        first_good_times (list, optional): Pre-computed first good meal times for
            each block. If provided with block_meal_info, avoids recomputing
            ML-based meal quality.

    Returns:
        pd.DataFrame: One row per block summarising transition percentages,
        first-meal timing, activity flags, and pellet rates.
    """
    new_add = []
    inactives = find_inactive_blocks(blocks, reverse=reverse)
    
    # If no pre-computed meal info, compute it now using block-based detection
    if block_meal_info is None:
        _, _, block_meal_info = find_meals_by_blocks(
            blocks,
            time_threshold=meal_config[0],
            pellet_threshold=meal_config[1],
            method=method,
        )

    for i, block in enumerate(blocks):
        no_pellet = remove_pellet(block)
        size = len(no_pellet)
        if size == 0:
            continue
            
        transitions = count_transitions(no_pellet)
        active_poke = block.iloc[0]['Active_Poke']

        times = block['Time'].tolist()
        block_time = round((times[-1] - times[0]).total_seconds() / 60, 2)
        
        # Use pre-computed block meals if available
        block_meals = block_meal_info[i]['meals'] if i < len(block_meal_info) else []
        time = round((block_meals[0][0] - times[0]).total_seconds() / 60, 2) if len(block_meals) > 0 else 'no meal'
        
        # Use pre-computed first good meal time if available
        if first_good_times is not None and i < len(first_good_times):
            first_meal_time = first_good_times[i]
        else:
            # Fallback: compute for this block
            _, first_meal_time = find_first_accurate_meal(block, 60, 2, 'cnn', method=method)
        
        if first_meal_time is None or first_meal_time > times[-1]:
            meal_1_good = block_time
        else:
            meal_1_good = round((first_meal_time - times[0]).total_seconds() / 60, 2)

        new_row_data = {
            'Block_Index': i+1,
            'Left_to_Left': round(transitions.get('Left_to_Left')/size * 100, 2),
            'Left_to_Right': round(transitions.get('Left_to_Right')/size * 100, 2),
            'Right_to_Right': round(transitions.get('Right_to_Right')/size * 100, 2),
            'Right_to_Left': round(transitions.get('Right_to_Left')/size * 100, 2),
            'Success_Count': transitions.get('success_count'),
            'Success_Rate' : round(transitions.get('success_count')/size * 100, 2),
            'Active_Poke' : active_poke,
            'First_Meal_Time': time,
            'First_Good_Meal_Time': min(meal_1_good, block_time),
            'Block_Time': block_time,
            'Incorrect_Pokes': size - transitions.get('success_count'),
            'Active': not (i in inactives)
        }
        new_add.append(new_row_data)

    idx = 0
    for each in new_add:
        count = count_pellet(blocks[idx])
        each['Pellet_Rate'] = round(count / len(blocks[idx]), 2)
        idx += 1
    
    data_stats = pd.DataFrame(new_add, columns=[
        'Block_Index', 'Left_to_Left', 'Left_to_Right', 'Right_to_Right', 'Right_to_Left',
        'Success_Count', 'Success_Rate','Active_Poke', 'First_Meal_Time', 'First_Good_Meal_Time',
        'Block_Time', 'Incorrect_Pokes', 'Active', 'Pellet_Rate'])

    return data_stats


def compute_session_analysis(
    data: pd.DataFrame,
    day_limit: int = 3,
    meal_config: tuple = (60, 2),
    method: str = 'paper',
    accuracy_threshold: float = 50.0,
) -> dict:
    """Compute blocks and meals together for a session to avoid redundant computation.
    
    This is the recommended entry point for reversal session analysis. It computes
    blocks once and detects meals within each block, ensuring no cross-block meals
    and avoiding multiple recomputation of meals.
    
    Args:
        data (pd.DataFrame): Raw session data with 'Time', 'Event', 'Active_Poke' columns.
        day_limit (int): Only include events within this many days from session start.
        meal_config (tuple): (time_threshold, pellet_threshold) for meal detection.
        method (str): Meal detection method ('paper' or 'ipi').
        accuracy_threshold (float): Minimum accuracy for meal acceptance.
    
    Returns:
        dict: Comprehensive session analysis results:
            - 'blocks': List of block DataFrames.
            - 'meal_analysis': Result from analyze_meals_by_blocks containing:
                - 'session_meals', 'session_meal_acc', 'block_meal_info'
                - 'meals_with_acc', 'good_mask', 'first_good_time'
                - 'in_meal_ratio', 'total_meals'
            - 'first_good_times_per_block': List of first good meal timestamps per block.
    """
    blocks = split_data_to_blocks(data, day=day_limit)
    
    if not blocks:
        return {
            'blocks': [],
            'meal_analysis': {
                'session_meals': [],
                'session_meal_acc': [],
                'block_meal_info': [],
                'meals_with_acc': [],
                'good_mask': np.zeros(0, dtype=bool),
                'first_good_time': None,
                'in_meal_ratio': 0.0,
                'total_meals': 0,
            },
            'first_good_times_per_block': [],
        }
    
    # Analyze meals by blocks (computes meals once for all blocks)
    meal_analysis = analyze_meals_by_blocks(
        blocks=blocks,
        time_threshold=meal_config[0],
        pellet_threshold=meal_config[1],
        model_type='cnn',
        accuracy_threshold=accuracy_threshold,
        method=method,
    )
    
    # Compute first good meal time per block for transition stats
    first_good_times_per_block = []
    block_meal_info = meal_analysis['block_meal_info']
    meals_with_acc = meal_analysis['meals_with_acc']
    good_mask = meal_analysis['good_mask']
    
    # Build a mapping of meal start times to their good_mask index
    meal_start_to_idx = {}
    for idx, (start_time, _) in enumerate(meals_with_acc):
        meal_start_to_idx[start_time] = idx
    
    for block_info in block_meal_info:
        block_meals = block_info['meals']
        first_good = None
        for meal_start, meal_end in block_meals:
            if meal_start in meal_start_to_idx:
                meal_idx = meal_start_to_idx[meal_start]
                if good_mask[meal_idx]:
                    first_good = pd.to_datetime(meal_start)
                    break
        first_good_times_per_block.append(first_good)
    
    return {
        'blocks': blocks,
        'meal_analysis': meal_analysis,
        'first_good_times_per_block': first_good_times_per_block,
    }


def first_meal_stats(data_stats: pd.DataFrame, ignore_inactive: bool = False) -> tuple[float, float, float]:
    """Summarise how quickly blocks reach their first good meal.

    Args:
        data_stats (pd.DataFrame): Output from ``get_transition_info``.
        ignore_inactive (bool): When True, drop blocks flagged as inactive
            before computing averages.

    Returns:
        tuple[float, float, float]: Mean ratio of first good meal duration to
        total block time, mean first meal latency, and median first good meal
        latency (all in minutes).
    """
    data_stats = data_stats[:-1]
    total_list = data_stats['Block_Time'].to_numpy(dtype=np.float32)
    time_list = np.array([time if type(time) == float else total_list[idx] 
                          for idx, time in enumerate(data_stats['First_Meal_Time'])])
    good_meal_list = np.array([time for time in data_stats['First_Good_Meal_Time']])

    if ignore_inactive:
        active_idx = [idx for idx, each in data_stats.iterrows() if each['Active']]
        time_list = time_list[active_idx]
        total_list = total_list[active_idx]
        good_meal_list = good_meal_list[active_idx]

    avg_ratio = np.mean(good_meal_list/total_list)
    avg_time = np.mean(time_list)
    avg_good_time = np.median(good_meal_list)
    return avg_ratio, avg_time, avg_good_time


def find_inactive_blocks(blocks: list[pd.DataFrame], reverse: bool) -> list[int]:
    """Identify which blocks fall in the nighttime portion of the cycle.

    Args:
        blocks (list[pd.DataFrame]): Blocked session events with ``Time`` column.
        reverse (bool): Swap the definition of inactive/active when True.

    Returns:
        list[int]: 1-based indices of blocks collected during inactive periods.
    """
    inactive_blocks = []
    block_start_index = 1

    for block_df in blocks:
        if not block_df.empty and 'Time' in block_df:
            times = pd.to_datetime(block_df['Time']).tolist()
            cnt = [1 if time.hour >= 19 or time.hour < 7 else 0 for time in times]
            if sum(cnt) > len(cnt) // 2:
                inactive_blocks.append(block_start_index)
        block_start_index += 1

    if reverse:
        inactive_blocks = [each for each in range(1, len(blocks)+1) if each not in inactive_blocks]
    return inactive_blocks


def plot_transition_stats(
    data_stats: pd.DataFrame,
    blocks: list[pd.DataFrame],
    *,
    mouse_label: str,
    group_label: str | None = None,
    export_path: str | os.PathLike | None = None,
    show: bool = False,
    inactive_reverse: bool = False,
) -> None:
    """Plot transition accuracy, success rates, and meal timing for one mouse.

    Args:
        data_stats (pd.DataFrame): Statistics returned by ``get_transition_info``.
        blocks (list[pd.DataFrame]): Original blocks for shading inactive spans.
        mouse_label (str): Identifier displayed in the title.
        group_label (str | None): Optional group name to show in the title.
        export_path (str | os.PathLike | None): When provided, save the figure
            to this path instead of only displaying it.
        show (bool): If True, display the figure immediately; otherwise close it
            after saving.
        inactive_reverse (bool): Pass-through flag to ``find_inactive_blocks``
            for toggling the shaded regions.
    """

    if data_stats.empty:
        return

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)

    block_idx = data_stats['Block_Index']
    transition_specs = [
        ('Left_to_Left', 'o', '#1f77b4', 'Left→Left'),
        ('Left_to_Right', '*', '#ff7f0e', 'Left→Right'),
        ('Right_to_Right', 's', '#2ca02c', 'Right→Right'),
        ('Right_to_Left', 'X', '#d62728', 'Right→Left'),
    ]

    line_handles = []
    for column, marker, color, label in transition_specs:
        if column not in data_stats:
            continue
        line, = ax.plot(
            block_idx,
            data_stats[column],
            marker=marker,
            color=color,
            linewidth=2,
            label=label,
        )
        line_handles.append(line)

    active_series = data_stats.get('Active_Poke', pd.Series(['Unknown'] * len(block_idx)))
    bar_palette = ['#f8b4c0' if str(poke).lower().startswith('l') else '#a4c8ff' for poke in active_series]
    success_rate = data_stats.get('Success_Rate')
    bars = None
    bars = ax.bar(block_idx, success_rate, color=bar_palette, alpha=0.6)

    first_good = data_stats['First_Good_Meal_Time'].tolist()
    block_time = data_stats['Block_Time'].tolist()

    for bar, meal_time, total_time in zip(bars, first_good, block_time):
        center_x = bar.get_x() + bar.get_width() / 2
        label_text = str(meal_time)
        total_text = f"{total_time}" if not np.isnan(total_time) else ""
        ax.text(center_x, bar.get_height() + 2.4, label_text, ha='center', va='bottom', fontsize=10)
        if total_text:
            ax.text(center_x, bar.get_height() + 0.6, total_text, ha='center', va='bottom', fontsize=10, color='#555555')

    plt.annotate('First Accurate Meal Time (min) \n Block Length', 
            xy=(bars[0].get_x()+0.4, bars[0].get_height() + 4.2), 
            xytext=(-90, 20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', lw=2),
            fontsize=16,
            color='blue')
    
    inactive_blocks = find_inactive_blocks(blocks, reverse=inactive_reverse)
    for block_index in inactive_blocks:
        ax.axvspan(block_index - 0.5, block_index + 0.5, facecolor='gray', alpha=0.25)

    legend_handles = line_handles.copy()
    if bars is not None:
        legend_handles.extend([
            mpatches.Patch(color='#f8b4c0', alpha=0.6, label='Left active'),
            mpatches.Patch(color='#a4c8ff', alpha=0.6, label='Right active'),
        ])
    if inactive_blocks:
        legend_handles.append(mpatches.Patch(color='gray', alpha=0.25, label='Inactive period'))
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=12, loc='upper right')

    title_parts = ['Transition Stats']
    if group_label:
        title_parts.append(f"Group {group_label}")
    title_parts.append(f"Mouse {mouse_label}")
    ax.set_title(' - '.join(title_parts), fontsize=20)

    ax.set_xlabel('Block Index', fontsize=14)
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_xticks(block_idx)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, linestyle='--')

    if export_path:
        fig.savefig(export_path, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)


def graph_tranition_stats(data_stats: pd.DataFrame, blocks: list[pd.DataFrame], sheet: str, export_path=None):
    """Backward-compatible alias used by earlier notebooks."""
    plot_transition_stats(
        data_stats,
        blocks,
        mouse_label=sheet,
        group_label=None,
        export_path=export_path,
        show=False,
    )
    return None


def wsls_from_transition_counts(transitions: dict[str, int], active_poke: str) -> tuple[float, float]:
    """Compute win-stay and lose-shift ratios from transition counts.

    Win-stay is P(stay | previous poke was correct).
    Lose-shift is P(shift | previous poke was incorrect).
    """
    win_stay_n, win_stay_d, lose_shift_n, lose_shift_d = wsls_count_components(
        transitions, active_poke
    )
    win_stay = (win_stay_n / win_stay_d) if win_stay_d > 0 else np.nan
    lose_shift = (lose_shift_n / lose_shift_d) if lose_shift_d > 0 else np.nan
    return win_stay, lose_shift


def wsls_count_components(transitions: dict[str, int], active_poke: str) -> tuple[int, int, int, int]:
    """Return WS/LS numerators and denominators for one block's transitions.

    Returns:
        tuple[int, int, int, int]:
            win_stay_n, win_stay_d, lose_shift_n, lose_shift_d
    """
    active_left = str(active_poke).lower().startswith('l')

    if active_left:
        win_stay_n = transitions.get('Left_to_Left', 0)
        win_stay_d = transitions.get('Left_to_Left', 0) + transitions.get('Left_to_Right', 0)
        lose_shift_n = transitions.get('Right_to_Left', 0)
        lose_shift_d = transitions.get('Right_to_Left', 0) + transitions.get('Right_to_Right', 0)
    else:
        win_stay_n = transitions.get('Right_to_Right', 0)
        win_stay_d = transitions.get('Right_to_Right', 0) + transitions.get('Right_to_Left', 0)
        lose_shift_n = transitions.get('Left_to_Right', 0)
        lose_shift_d = transitions.get('Left_to_Right', 0) + transitions.get('Left_to_Left', 0)

    return win_stay_n, win_stay_d, lose_shift_n, lose_shift_d


def wsls_for_block(block: pd.DataFrame) -> tuple[float, float]:
    """Compute WSLS ratios for one block using existing transition patterns."""
    no_pellet = remove_pellet(block)
    if len(no_pellet) < 2:
        return np.nan, np.nan

    transitions = count_transitions(no_pellet)
    active_poke = no_pellet.iloc[0]['Active_Poke']
    return wsls_from_transition_counts(transitions, active_poke)


def wsls_for_session_blocks(blocks: list[pd.DataFrame]) -> tuple[float, float]:
    """Compute WSLS for one subject/session using pooled transitions.

    This sums transition counts across all blocks first, then computes
    win-stay and lose-shift ratios once per subject/session.
    """
    win_stay_n_total = 0
    win_stay_d_total = 0
    lose_shift_n_total = 0
    lose_shift_d_total = 0

    for block in blocks:
        no_pellet = remove_pellet(block)
        if len(no_pellet) < 2:
            continue

        transitions = count_transitions(no_pellet)
        active_poke = no_pellet.iloc[0]['Active_Poke']
        ws_n, ws_d, ls_n, ls_d = wsls_count_components(transitions, active_poke)

        win_stay_n_total += ws_n
        win_stay_d_total += ws_d
        lose_shift_n_total += ls_n
        lose_shift_d_total += ls_d

    win_stay = (win_stay_n_total / win_stay_d_total) if win_stay_d_total > 0 else np.nan
    lose_shift = (lose_shift_n_total / lose_shift_d_total) if lose_shift_d_total > 0 else np.nan
    return win_stay, lose_shift


def _block_slice_by_pellet_window(
    block: pd.DataFrame,
    pellet_window: int,
    window: str,
) -> pd.DataFrame | None:
    """Slice one block by first/last N pellet events.

    Args:
        block (pd.DataFrame): Full block containing poke and pellet events.
        pellet_window (int): Number of pellets to include for slicing.
        window (str): Either ``'first'`` or ``'last'``.

    Returns:
        pd.DataFrame | None: Sliced block (reset index) or ``None`` if the block
        has fewer than ``pellet_window`` pellets.
    """
    if pellet_window <= 0:
        raise ValueError("pellet_window must be a positive integer.")
    if window not in {'first', 'last'}:
        raise ValueError("window must be either 'first' or 'last'.")

    pellet_indices = np.flatnonzero(block['Event'].to_numpy() == 'Pellet')
    if pellet_indices.size < pellet_window:
        return None

    if window == 'first':
        end_idx = int(pellet_indices[pellet_window - 1])
        sliced = block.iloc[:end_idx + 1]
    else:
        start_idx = int(pellet_indices[-pellet_window])
        sliced = block.iloc[start_idx:]

    return sliced.reset_index(drop=True)


def wsls_for_session_blocks_pellet_windows(
    blocks: list[pd.DataFrame],
    pellet_window: int = 10,
    *,
    exclude_last_block: bool = True,
) -> dict[str, float]:
    """Compute per-session WSLS for first/last pellet windows.

    Workflow:
        1) Compute WSLS per block on first/last ``pellet_window`` pellets.
        2) Average block-level ratios within each session.
        3) Optionally exclude the final block (recommended for incomplete tails).

    Args:
        blocks (list[pd.DataFrame]): Session blocks in chronological order.
        pellet_window (int): Number of pellet events used for each edge window.
        exclude_last_block (bool): If True, drop the final block before analysis.

    Returns:
        dict[str, float]: Session means for each strategy/window combination:
            - ``win_stay_first``
            - ``lose_shift_first``
            - ``win_stay_last``
            - ``lose_shift_last``
    """
    if exclude_last_block:
        blocks_to_use = blocks[:-1]
    else:
        blocks_to_use = blocks

    win_stay_first_vals: list[float] = []
    lose_shift_first_vals: list[float] = []
    win_stay_last_vals: list[float] = []
    lose_shift_last_vals: list[float] = []

    for block in blocks_to_use:
        first_slice = _block_slice_by_pellet_window(block, pellet_window, window='first')
        if first_slice is not None:
            ws_first, ls_first = wsls_for_block(first_slice)
            if np.isfinite(ws_first):
                win_stay_first_vals.append(float(ws_first))
            if np.isfinite(ls_first):
                lose_shift_first_vals.append(float(ls_first))

        last_slice = _block_slice_by_pellet_window(block, pellet_window, window='last')
        if last_slice is not None:
            ws_last, ls_last = wsls_for_block(last_slice)
            if np.isfinite(ws_last):
                win_stay_last_vals.append(float(ws_last))
            if np.isfinite(ls_last):
                lose_shift_last_vals.append(float(ls_last))

    return {
        'win_stay_first': float(np.mean(win_stay_first_vals)) if win_stay_first_vals else np.nan,
        'lose_shift_first': float(np.mean(lose_shift_first_vals)) if lose_shift_first_vals else np.nan,
        'win_stay_last': float(np.mean(win_stay_last_vals)) if win_stay_last_vals else np.nan,
        'lose_shift_last': float(np.mean(lose_shift_last_vals)) if lose_shift_last_vals else np.nan,
    }


def wsls_pellet_window_by_group(
    blocks_by_group: dict[str, list[list[pd.DataFrame]]],
    pellet_window: int = 10,
    *,
    exclude_last_block: bool = True,
    verbose: bool = True,
    return_summary_df: bool = False,
) -> dict[str, dict[str, list[float]]] | tuple[dict[str, dict[str, list[float]]], pd.DataFrame]:
    """Aggregate first/last-pellet WSLS per group.

    Session values are first averaged within session across blocks (excluding the
    final block by default), then aggregated at the group level.

    Args:
        blocks_by_group (dict[str, list[list[pd.DataFrame]]]): Mapping
            ``group -> list of sessions``, where each session is a list of blocks.
        pellet_window (int): Number of pellets for each edge window.
        exclude_last_block (bool): Whether to omit the final block per session.
        verbose (bool): If True, print Mean/SE table by group, window, strategy.
        return_summary_df (bool): If True, also return a tidy summary DataFrame.

    Returns:
        dict[str, dict[str, list[float]]] | tuple[...]:
            Always returns the four per-group metric dictionaries:
                - ``win_stay_first``
                - ``lose_shift_first``
                - ``win_stay_last``
                - ``lose_shift_last``
            If ``return_summary_df`` is True, also returns a summary DataFrame
            with columns ``Group, Block_Edge, Strategy, n, Mean, SE``.
    """
    metric_names = [
        'win_stay_first',
        'lose_shift_first',
        'win_stay_last',
        'lose_shift_last',
    ]
    wsls_metrics = {
        metric_name: {group: [] for group in blocks_by_group}
        for metric_name in metric_names
    }

    for group, group_sessions in blocks_by_group.items():
        for session_blocks in group_sessions:
            session_metrics = wsls_for_session_blocks_pellet_windows(
                session_blocks,
                pellet_window=pellet_window,
                exclude_last_block=exclude_last_block,
            )
            for metric_name in metric_names:
                metric_val = session_metrics[metric_name]
                if np.isfinite(metric_val):
                    wsls_metrics[metric_name][group].append(float(metric_val))

    summary_rows: list[dict[str, str | int | float]] = []
    strategy_map = {'win_stay': 'win-stay', 'lose_shift': 'lose-shift'}
    window_map = {'first': f'first {pellet_window}', 'last': f'last {pellet_window}'}
    for group in blocks_by_group:
        for window_key in ['first', 'last']:
            for strategy_key in ['win_stay', 'lose_shift']:
                metric_key = f'{strategy_key}_{window_key}'
                mean_val, sem_val, n_val = _mean_sem(wsls_metrics[metric_key].get(group, []))
                summary_rows.append(
                    {
                        'Group': group,
                        'Block_Edge': window_map[window_key],
                        'Strategy': strategy_map[strategy_key],
                        'n': int(n_val),
                        'Mean': mean_val,
                        'SE': sem_val,
                    }
                )
    summary_df = pd.DataFrame(summary_rows, columns=['Group', 'Block_Edge', 'Strategy', 'n', 'Mean', 'SE'])

    if verbose:
        print(f"[WSLS] Mean and SE by group, strategy, and block edge ({pellet_window} pellets)")
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if return_summary_df:
        return wsls_metrics, summary_df
    return wsls_metrics


def wsls_pellet_window_from_session_analyses(
    session_analyses_by_group: dict[str, list[dict]],
    pellet_window: int = 10,
    *,
    exclude_last_block: bool = True,
    verbose: bool = True,
    return_summary_df: bool = False,
    plot: bool = True,
    export_path: str | os.PathLike | None = None,
    show: bool = True,
) -> dict[str, dict[str, list[float]]] | tuple[dict[str, dict[str, list[float]]], pd.DataFrame]:
    """Convenience wrapper for ``compute_session_analysis`` outputs.

    Args:
        session_analyses_by_group (dict[str, list[dict]]): Mapping
            ``group -> list of analysis dicts`` that contain ``'blocks'``.
        pellet_window (int): Number of pellets for first/last windows.
        exclude_last_block (bool): Whether to omit each session's final block.
        verbose (bool): If True, print Mean/SE table.
        return_summary_df (bool): If True, also return a tidy summary DataFrame.
        plot (bool): If True, render first/last pellet-window WSLS violin plots.
        export_path (str | os.PathLike | None): Optional output path for the figure.
        show (bool): If True, display the figure; otherwise close after saving.

    Returns:
        dict[str, dict[str, list[float]]] | tuple[...]: Output of
        ``wsls_pellet_window_by_group``.
    """
    blocks_by_group = {
        group: [analysis.get('blocks', []) for analysis in analyses]
        for group, analyses in session_analyses_by_group.items()
    }

    wsls_metrics, summary_df = wsls_pellet_window_by_group(
        blocks_by_group=blocks_by_group,
        pellet_window=pellet_window,
        exclude_last_block=exclude_last_block,
        verbose=verbose,
        return_summary_df=True,
    )

    if plot and blocks_by_group:
        strategy_specs = [
            ('win_stay', 'win-stay', palette[0]),
            ('lose_shift', 'lose-shift', palette[1]),
        ]
        window_specs = [
            ('first', f'first {pellet_window} pellets'),
            ('last', f'last {pellet_window} pellets'),
        ]
        group_labels = list(blocks_by_group.keys())

        fig_width = max(12, 3.2 * max(1, len(group_labels)))
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5.6), sharey=True)

        for ax, (window_key, window_title) in zip(axes, window_specs):
            violin_data: list[np.ndarray] = []
            violin_positions: list[float] = []
            violin_colors: list[str] = []
            tick_positions: list[float] = []
            tick_labels: list[str] = []
            group_centers: list[tuple[float, str]] = []

            x_cursor = 0.0
            within_group_gap = 0.9
            group_gap = 1.5

            for group in group_labels:
                ws_vals = _clean_numeric_values(wsls_metrics[f'win_stay_{window_key}'].get(group, []))
                ls_vals = _clean_numeric_values(wsls_metrics[f'lose_shift_{window_key}'].get(group, []))
                ws_pos = x_cursor
                ls_pos = x_cursor + within_group_gap

                tick_positions.extend([ws_pos, ls_pos])
                tick_labels.extend(['WS', 'LS'])
                group_centers.append(((ws_pos + ls_pos) / 2.0, group))

                if ws_vals.size > 0:
                    violin_data.append(ws_vals)
                    violin_positions.append(ws_pos)
                    violin_colors.append(strategy_specs[0][2])
                if ls_vals.size > 0:
                    violin_data.append(ls_vals)
                    violin_positions.append(ls_pos)
                    violin_colors.append(strategy_specs[1][2])

                x_cursor += within_group_gap + group_gap

            if violin_data:
                violin_parts = ax.violinplot(
                    violin_data,
                    positions=violin_positions,
                    widths=0.55,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for idx, body in enumerate(violin_parts['bodies']):
                    body.set_facecolor(violin_colors[idx])
                    body.set_edgecolor('black')
                    body.set_alpha(0.65)

                ax.boxplot(
                    violin_data,
                    positions=violin_positions,
                    widths=0.28,
                    showfliers=False,
                    patch_artist=True,
                    boxprops=dict(facecolor='white', edgecolor='black'),
                    medianprops=dict(color='black'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                )

                for pos, values, color in zip(violin_positions, violin_data, violin_colors):
                    jitter = np.random.uniform(-0.08, 0.08, size=len(values))
                    ax.scatter(
                        np.repeat(pos, len(values)) + jitter,
                        values,
                        color=color,
                        edgecolor='black',
                        linewidth=0.4,
                        alpha=0.85,
                        zorder=3,
                    )

                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)
                for idx, (center, group) in enumerate(group_centers):
                    ax.text(
                        center,
                        -0.15,
                        group,
                        transform=ax.get_xaxis_transform(),
                        ha='center',
                        va='top',
                        fontsize=11,
                        fontweight='bold',
                    )
                    if idx < len(group_centers) - 1:
                        sep_x = (group_centers[idx][0] + group_centers[idx + 1][0]) / 2.0
                        ax.axvline(sep_x, color='black', linewidth=0.6, alpha=0.2)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])

            ax.set_title(window_title)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.25)

        legend_handles = [
            mpatches.Patch(color=strategy_specs[0][2], alpha=0.65, label='win-stay'),
            mpatches.Patch(color=strategy_specs[1][2], alpha=0.65, label='lose-shift'),
        ]
        axes[0].legend(handles=legend_handles, fontsize=9, frameon=False, loc='lower right')
        axes[0].set_ylabel('Probability')
        fig.suptitle(f'WSLS by group: first vs last {pellet_window} pellets', y=0.98)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        if export_path is not None:
            export_path_str = os.fspath(export_path)
            export_dir = os.path.dirname(export_path_str)
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)
            fig.savefig(export_path_str, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

    if return_summary_df:
        return wsls_metrics, summary_df
    return wsls_metrics


def session_poke_accuracy(blocks: list[pd.DataFrame]) -> float:
    """Compute session-level poke accuracy (global, not block-averaged).

    Only Left/Right poke events are considered (pellet and other events ignored).
    """
    total_events = 0
    correct_events = 0

    for block in blocks:
        poke_rows = block[block['Event'].isin(['Left', 'Right'])]
        if poke_rows.empty:
            continue
        total_events += len(poke_rows)
        correct_events += int((poke_rows['Event'] == poke_rows['Active_Poke']).sum())

    if total_events == 0:
        return np.nan
    return (correct_events / total_events) * 100


def _mean_sem(values: list[float]) -> tuple[float, float, int]:
    """Return mean, SEM, and sample size."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem, n


def _find_group_by_alias(group_labels: list[str], aliases: set[str]) -> str | None:
    """Find first group whose lowercase name matches a provided alias."""
    for group in group_labels:
        if group.lower() in aliases:
            return group
    return None


def _clean_numeric_values(values: list[float]) -> np.ndarray:
    """Convert a sequence to finite float values only."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _plot_violin_scatter_panel(
    ax,
    datasets: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    title: str,
    legend_labels: list[str] | None = None,
) -> None:
    """Render violin + box + scatter panel for WSLS values."""
    if not datasets:
        ax.set_visible(False)
        return

    positions = np.arange(len(datasets))
    violin_parts = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.55,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for idx, body in enumerate(violin_parts['bodies']):
        body.set_facecolor(colors[idx])
        body.set_edgecolor('black')
        body.set_alpha(0.65)

    ax.boxplot(
        datasets,
        positions=positions,
        widths=0.28,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='white', edgecolor='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )

    for idx, values in enumerate(datasets):
        jitter = np.random.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.repeat(positions[idx], len(values)) + jitter,
            values,
            color=colors[idx],
            edgecolor='black',
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )

    if legend_labels is not None:
        legend_handles = [
            mpatches.Patch(color=colors[idx], alpha=0.65, label=legend_labels[idx])
            for idx in range(len(legend_labels))
        ]
        ax.legend(handles=legend_handles, fontsize=9, frameon=False, loc='lower right')

    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.25)


def _format_mean_se(values: np.ndarray) -> tuple[float, float]:
    """Return mean and standard error for one strategy distribution."""
    n = int(values.size)
    if n == 0:
        return np.nan, np.nan
    mean_val = float(np.mean(values))
    se_val = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean_val, se_val


def plot_wsls_two_panel(
    win_stay_by_group: dict[str, list[float]],
    lose_shift_by_group: dict[str, list[float]],
    *,
    group_labels: list[str],
    export_path: str | os.PathLike | None = None,
    show: bool = True,
) -> None:
    """Plot 1x2 WSLS comparison (WT panel + ctrl/cask grouped panel).

    Both panels use violin + box + jittered scatter points to align with the
    visual style used by ``plot_group_stats_wrapper``.
    """
    wt_group = _find_group_by_alias(group_labels, {'female'})
    ctrl_group = _find_group_by_alias(group_labels, {'ctrl', 'control'})
    cask_group = _find_group_by_alias(group_labels, {'cask'})

    strategy_colors = [palette[0], palette[1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    stats_report_rows: list[tuple[str, str, int, float, float]] = []

    # Left panel: WT
    if wt_group is not None:
        wt_ws = _clean_numeric_values(win_stay_by_group.get(wt_group, []))
        wt_ls = _clean_numeric_values(lose_shift_by_group.get(wt_group, []))

        left_data: list[np.ndarray] = []
        left_labels: list[str] = []
        left_colors: list[str] = []
        if wt_ws.size > 0:
            left_data.append(wt_ws)
            left_labels.append('win-stay')
            left_colors.append(strategy_colors[0])
            wt_ws_mean, wt_ws_se = _format_mean_se(wt_ws)
            stats_report_rows.append((wt_group, 'win-stay', int(wt_ws.size), wt_ws_mean, wt_ws_se))
        if wt_ls.size > 0:
            left_data.append(wt_ls)
            left_labels.append('lose-shift')
            left_colors.append(strategy_colors[1])
            wt_ls_mean, wt_ls_se = _format_mean_se(wt_ls)
            stats_report_rows.append((wt_group, 'lose-shift', int(wt_ls.size), wt_ls_mean, wt_ls_se))

        if left_data:
            wt_n = max(int(wt_ws.size), int(wt_ls.size))
            _plot_violin_scatter_panel(
                axes[0],
                left_data,
                left_labels,
                left_colors,
                title=f'{wt_group} mice (n={wt_n})',
                legend_labels=left_labels,
            )
        else:
            axes[0].text(0.5, 0.5, 'No WT WSLS data', ha='center', va='center', fontsize=12)
            axes[0].set_xticks([])
            axes[0].set_title(f'{wt_group} mice')
    else:
        axes[0].text(0.5, 0.5, 'WT group not found', ha='center', va='center', fontsize=12)
        axes[0].set_xticks([])
        axes[0].set_title('WT mice')

    # Right panel: ctrl vs cask (or best available pair)
    comparison_groups = [g for g in [ctrl_group, cask_group] if g is not None]
    if len(comparison_groups) < 2:
        fallback_groups = [g for g in group_labels if g not in comparison_groups and g != wt_group]
        for candidate in fallback_groups:
            if len(comparison_groups) >= 2:
                break
            comparison_groups.append(candidate)

    if comparison_groups:
        ax = axes[1]
        violin_data: list[np.ndarray] = []
        violin_positions: list[float] = []
        violin_colors: list[str] = []
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        group_centers: list[tuple[float, str]] = []
        group_n_map: dict[str, int] = {}

        x_cursor = 0.0
        group_gap = 1.4
        within_group_gap = 0.9

        for group in comparison_groups:
            ws_vals = _clean_numeric_values(win_stay_by_group.get(group, []))
            ls_vals = _clean_numeric_values(lose_shift_by_group.get(group, []))
            ws_pos = x_cursor
            ls_pos = x_cursor + within_group_gap

            tick_positions.extend([ws_pos, ls_pos])
            tick_labels.extend(['WS', 'LS'])
            group_centers.append(((ws_pos + ls_pos) / 2.0, group))
            group_n_map[group] = max(int(ws_vals.size), int(ls_vals.size))

            if ws_vals.size > 0:
                violin_data.append(ws_vals)
                violin_positions.append(ws_pos)
                violin_colors.append(strategy_colors[0])
                ws_mean, ws_se = _format_mean_se(ws_vals)
                stats_report_rows.append((group, 'win-stay', int(ws_vals.size), ws_mean, ws_se))

            if ls_vals.size > 0:
                violin_data.append(ls_vals)
                violin_positions.append(ls_pos)
                violin_colors.append(strategy_colors[1])
                ls_mean, ls_se = _format_mean_se(ls_vals)
                stats_report_rows.append((group, 'lose-shift', int(ls_vals.size), ls_mean, ls_se))

            x_cursor += within_group_gap + group_gap

        if violin_data:
            violin_parts = ax.violinplot(
                violin_data,
                positions=violin_positions,
                widths=0.55,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for idx, body in enumerate(violin_parts['bodies']):
                body.set_facecolor(violin_colors[idx])
                body.set_edgecolor('black')
                body.set_alpha(0.65)

            ax.boxplot(
                violin_data,
                positions=violin_positions,
                widths=0.28,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor='white', edgecolor='black'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
            )

            for pos, values, color in zip(violin_positions, violin_data, violin_colors):
                jitter = np.random.uniform(-0.08, 0.08, size=len(values))
                ax.scatter(
                    np.repeat(pos, len(values)) + jitter,
                    values,
                    color=color,
                    edgecolor='black',
                    linewidth=0.4,
                    alpha=0.85,
                    zorder=3,
                )

            legend_handles = [
                mpatches.Patch(color=strategy_colors[0], alpha=0.65, label='win-stay'),
                mpatches.Patch(color=strategy_colors[1], alpha=0.65, label='lose-shift'),
            ]
            ax.legend(handles=legend_handles, fontsize=9, frameon=False, loc='lower right')

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.25)

            for idx, (center, group) in enumerate(group_centers):
                ax.text(
                    center,
                    -0.14,
                    group,
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='top',
                    fontsize=12,
                    fontweight='bold',
                )
                if idx < len(group_centers) - 1:
                    sep_x = (group_centers[idx][0] + group_centers[idx + 1][0]) / 2.0
                    ax.axvline(sep_x, color='black', linewidth=0.6, alpha=0.25)

            if ctrl_group in group_n_map and cask_group in group_n_map:
                ax.set_title(
                    f'ctrl (n={group_n_map[ctrl_group]}) vs. cask (n={group_n_map[cask_group]})'
                )
            elif len(comparison_groups) >= 2:
                g0, g1 = comparison_groups[:2]
                ax.set_title(f'{g0} (n={group_n_map[g0]}) vs. {g1} (n={group_n_map[g1]})')
            else:
                g0 = comparison_groups[0]
                ax.set_title(f'{g0} (n={group_n_map[g0]})')
        else:
            axes[1].text(0.5, 0.5, 'No WSLS data for comparison groups', ha='center', va='center', fontsize=12)
            axes[1].set_xticks([])
            axes[1].set_title('Group comparison')
    else:
        axes[1].text(0.5, 0.5, 'ctrl/cask groups not found', ha='center', va='center', fontsize=12)
        axes[1].set_xticks([])
        axes[1].set_title('Group comparison')

    axes[0].set_ylabel('Ratio')
    fig.suptitle('Reversal strategies: win-stay and lose-shift', fontsize=13)
    fig.tight_layout()

    if export_path:
        fig.savefig(export_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    if stats_report_rows:
        print("[WSLS] Mean and SE by group and strategy")
        print("Group\tStrategy\tn\tMean\tSE")
        for group_name, strategy_name, n_val, mean_val, se_val in stats_report_rows:
            print(f"{group_name}\t{strategy_name}\t{n_val}\t{mean_val:.4f}\t{se_val:.4f}")


def run_wsls_within_group_strategy_tests(
    win_stay_by_group: dict[str, list[float]],
    lose_shift_by_group: dict[str, list[float]],
    *,
    group_order: list[str] | None = None,
) -> None:
    """Run WS vs LS two-sided tests within each group.

    This mirrors the configuration used by ``run_pairwise_tests`` (independent,
    two-sided t-test), but compares strategies inside the same group.
    """
    if group_order is None:
        groups = [group for group in win_stay_by_group.keys() if group in lose_shift_by_group]
    else:
        groups = group_order

    if not groups:
        raise ValueError("No groups available for WS vs LS testing.")

    missing_groups = [
        group for group in groups
        if group not in win_stay_by_group or group not in lose_shift_by_group
    ]
    if missing_groups:
        raise KeyError(f"Missing WSLS data for groups: {missing_groups}")

    print("[WSLS] Within-group strategy tests (win-stay vs lose-shift)")
    for group in groups:
        strategy_map = {
            'win-stay': win_stay_by_group[group],
            'lose-shift': lose_shift_by_group[group],
        }
        run_pairwise_tests(
            strategy_map,
            metric_name=f"WS vs LS ({group})",
            cohort_pairs=[('win-stay', 'lose-shift')],
        )


def run_wsls_first_last_window_tests(
    wsls_window_metrics: dict[str, dict[str, list[float]]],
    *,
    pellet_window: int = 10,
    group_order: list[str] | None = None,
) -> None:
    """Run first-vs-last pellet-window tests per group and strategy.

    For each group, this performs:
      1) win-stay: first N vs last N
      2) lose-shift: first N vs last N
    where ``N`` is ``pellet_window``.
    """
    required_keys = ['win_stay_first', 'win_stay_last', 'lose_shift_first', 'lose_shift_last']
    missing_keys = [metric_key for metric_key in required_keys if metric_key not in wsls_window_metrics]
    if missing_keys:
        raise KeyError(f"Missing WSLS window metrics: {missing_keys}")

    if group_order is None:
        groups = list(wsls_window_metrics['win_stay_first'].keys())
    else:
        groups = group_order

    if not groups:
        raise ValueError("No groups available for first/last WSLS testing.")

    missing_groups = []
    for group in groups:
        for metric_key in required_keys:
            if group not in wsls_window_metrics[metric_key]:
                missing_groups.append((group, metric_key))
    if missing_groups:
        raise KeyError(f"Missing group metrics for first/last WSLS tests: {missing_groups}")

    first_label = f'first {pellet_window}'
    last_label = f'last {pellet_window}'
    print(f"[WSLS] First-vs-last {pellet_window} pellet tests by group and strategy")

    for group in groups:
        win_stay_map = {
            first_label: wsls_window_metrics['win_stay_first'][group],
            last_label: wsls_window_metrics['win_stay_last'][group],
        }
        run_pairwise_tests(
            win_stay_map,
            metric_name=f"Win-stay first/last ({group})",
            cohort_pairs=[(first_label, last_label)],
        )

        lose_shift_map = {
            first_label: wsls_window_metrics['lose_shift_first'][group],
            last_label: wsls_window_metrics['lose_shift_last'][group],
        }
        run_pairwise_tests(
            lose_shift_map,
            metric_name=f"Lose-shift first/last ({group})",
            cohort_pairs=[(first_label, last_label)],
        )


def _holm_bonferroni_adjust(p_values: list[float]) -> np.ndarray:
    """Adjust p-values with Holm-Bonferroni correction."""
    p_arr = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_arr, np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(p_arr))
    if valid_idx.size == 0:
        return adjusted

    p_valid = p_arr[valid_idx]
    order = np.argsort(p_valid)
    p_sorted = p_valid[order]
    m = p_sorted.size

    scaled = np.array([(m - i) * p_sorted[i] for i in range(m)], dtype=float)
    adjusted_sorted = np.maximum.accumulate(scaled)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    adjusted_valid = adjusted_sorted[inv_order]
    adjusted[valid_idx] = adjusted_valid
    return adjusted


def _adjust_pvalues(p_values: list[float], method: str = 'holm') -> np.ndarray:
    """Adjust p-values using a simple built-in correction method."""
    method = str(method).lower()
    p_arr = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_arr, np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(p_arr))
    if valid_idx.size == 0:
        return adjusted

    p_valid = p_arr[valid_idx]
    m = p_valid.size

    if method == 'holm':
        adjusted_valid = _holm_bonferroni_adjust(p_valid.tolist())
    elif method == 'bonferroni':
        adjusted_valid = np.clip(p_valid * m, 0.0, 1.0)
    elif method in {'fdr_bh', 'bh'}:
        order = np.argsort(p_valid)
        p_sorted = p_valid[order]
        bh = np.array([p_sorted[i] * m / (i + 1) for i in range(m)], dtype=float)
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        bh = np.clip(bh, 0.0, 1.0)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(m)
        adjusted_valid = bh[inv_order]
    else:
        raise ValueError("Unsupported correction method. Use 'holm', 'bonferroni', or 'fdr_bh'.")

    adjusted[valid_idx] = adjusted_valid
    return adjusted


def _fit_ols_sse(y: np.ndarray, x: np.ndarray) -> tuple[float, int, int, np.ndarray]:
    """Fit OLS and return SSE, df_error, rank, and beta."""
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - (x @ beta)
    sse = float(np.sum(residuals ** 2))
    n_obs = int(y.shape[0])
    df_error = int(n_obs - rank)
    return sse, df_error, int(rank), beta


def _build_two_level_code(
    series: pd.Series,
    level_order: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build effect-coding (-0.5, +0.5) for a two-level factor."""
    values = series.astype(str)
    levels = [str(v) for v in pd.unique(values)]
    if level_order is not None:
        level_order_str = [str(v) for v in level_order if str(v) in levels]
        if len(level_order_str) != 2:
            raise ValueError("level_order must contain exactly two present levels.")
        levels = level_order_str
    else:
        levels = sorted(levels)
        if len(levels) != 2:
            raise ValueError("Two-way ANOVA helper currently supports exactly two levels per factor.")

    code_map = {levels[0]: -0.5, levels[1]: 0.5}
    coded = np.array([code_map[v] for v in values], dtype=float)
    return coded, levels


def _run_two_way_anova_with_posthoc(
    data: pd.DataFrame,
    *,
    value_col: str,
    factor_a_col: str,
    factor_b_col: str,
    factor_a_order: list[str] | None = None,
    factor_b_order: list[str] | None = None,
    correction_method: str = 'holm',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a 2x2 fixed-effects ANOVA (with interaction) and posthoc pairwise tests."""
    required_cols = [value_col, factor_a_col, factor_b_col]
    df = data[required_cols].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise ValueError("No valid rows available for ANOVA.")

    xa, factor_a_levels = _build_two_level_code(df[factor_a_col], level_order=factor_a_order)
    xb, factor_b_levels = _build_two_level_code(df[factor_b_col], level_order=factor_b_order)
    y = df[value_col].to_numpy(dtype=float)

    x_full = np.column_stack([np.ones_like(y), xa, xb, xa * xb])
    sse_full, df_error, _, beta = _fit_ols_sse(y, x_full)
    if df_error <= 0:
        raise ValueError("Insufficient residual degrees of freedom for ANOVA.")

    mse_error = sse_full / df_error
    term_specs = [
        ('group', 1),
        ('strategy_or_window', 2),
        ('interaction', 3),
    ]
    term_rename = {
        'group': factor_a_col,
        'strategy_or_window': factor_b_col,
        'interaction': f"{factor_a_col}:{factor_b_col}",
    }

    anova_rows: list[dict[str, float | str | int]] = []
    for term_key, col_idx in term_specs:
        x_reduced = np.delete(x_full, col_idx, axis=1)
        sse_reduced, _, _, _ = _fit_ols_sse(y, x_reduced)
        ss_term = max(0.0, sse_reduced - sse_full)
        f_stat = (ss_term / 1.0) / mse_error if mse_error > 0 else np.nan
        p_value = float(stats.f.sf(f_stat, 1, df_error)) if np.isfinite(f_stat) else np.nan
        anova_rows.append(
            {
                'Term': term_rename[term_key],
                'df': 1,
                'SS': ss_term,
                'MS': ss_term,
                'F': f_stat,
                'p_value': p_value,
            }
        )

    total_ss = float(np.sum((y - np.mean(y)) ** 2))
    residual_ss = float(sse_full)
    r_squared = 1.0 - (residual_ss / total_ss) if total_ss > 0 else np.nan
    anova_rows.append(
        {
            'Term': 'Residual',
            'df': int(df_error),
            'SS': residual_ss,
            'MS': mse_error,
            'F': np.nan,
            'p_value': np.nan,
        }
    )
    anova_df = pd.DataFrame(anova_rows, columns=['Term', 'df', 'SS', 'MS', 'F', 'p_value'])

    # Build posthoc comparisons
    result_rows: list[dict[str, float | str | int]] = []

    def _append_ttest(
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        *,
        family: str,
        label_1: str,
        label_2: str,
    ) -> None:
        arr1 = sample_1[np.isfinite(sample_1)]
        arr2 = sample_2[np.isfinite(sample_2)]
        if arr1.size == 0 or arr2.size == 0:
            t_stat = np.nan
            p_raw = np.nan
        else:
            t_stat, p_raw = stats.ttest_ind(arr1, arr2, equal_var=False, nan_policy='omit')
        result_rows.append(
            {
                'family': family,
                'comparison': f"{label_1} vs {label_2}",
                'label_1': label_1,
                'label_2': label_2,
                'n_1': int(arr1.size),
                'n_2': int(arr2.size),
                'mean_1': float(np.mean(arr1)) if arr1.size > 0 else np.nan,
                'mean_2': float(np.mean(arr2)) if arr2.size > 0 else np.nan,
                't_stat': float(t_stat) if np.isfinite(t_stat) else np.nan,
                'p_raw': float(p_raw) if np.isfinite(p_raw) else np.nan,
            }
        )

    # Cell-level pairwise: all 2x2 combinations
    cell_data: dict[str, np.ndarray] = {}
    for a_level in factor_a_levels:
        for b_level in factor_b_levels:
            mask = (df[factor_a_col].astype(str) == str(a_level)) & (df[factor_b_col].astype(str) == str(b_level))
            label = f"{a_level} | {b_level}"
            cell_data[label] = df.loc[mask, value_col].to_numpy(dtype=float)
    for left, right in combinations(cell_data.keys(), 2):
        _append_ttest(cell_data[left], cell_data[right], family='cell_pairwise', label_1=left, label_2=right)

    # Simple effects of factor A within each level of factor B
    for b_level in factor_b_levels:
        samples_by_a = []
        labels_by_a = []
        for a_level in factor_a_levels:
            mask = (df[factor_a_col].astype(str) == str(a_level)) & (df[factor_b_col].astype(str) == str(b_level))
            samples_by_a.append(df.loc[mask, value_col].to_numpy(dtype=float))
            labels_by_a.append(f"{factor_a_col}={a_level} @ {factor_b_col}={b_level}")
        _append_ttest(
            samples_by_a[0],
            samples_by_a[1],
            family=f'simple_{factor_a_col}_within_{factor_b_col}',
            label_1=labels_by_a[0],
            label_2=labels_by_a[1],
        )

    # Simple effects of factor B within each level of factor A
    for a_level in factor_a_levels:
        samples_by_b = []
        labels_by_b = []
        for b_level in factor_b_levels:
            mask = (df[factor_a_col].astype(str) == str(a_level)) & (df[factor_b_col].astype(str) == str(b_level))
            samples_by_b.append(df.loc[mask, value_col].to_numpy(dtype=float))
            labels_by_b.append(f"{factor_b_col}={b_level} @ {factor_a_col}={a_level}")
        _append_ttest(
            samples_by_b[0],
            samples_by_b[1],
            family=f'simple_{factor_b_col}_within_{factor_a_col}',
            label_1=labels_by_b[0],
            label_2=labels_by_b[1],
        )

    posthoc_df = pd.DataFrame(
        result_rows,
        columns=[
            'family', 'comparison', 'label_1', 'label_2', 'n_1', 'n_2',
            'mean_1', 'mean_2', 't_stat', 'p_raw',
        ],
    )
    posthoc_df['p_adj'] = _adjust_pvalues(posthoc_df['p_raw'].tolist(), method=correction_method)
    posthoc_df['correction'] = correction_method

    # Add fit summary columns
    anova_df.attrs['factor_a_levels'] = factor_a_levels
    anova_df.attrs['factor_b_levels'] = factor_b_levels
    anova_df.attrs['coefficients'] = {
        'Intercept': float(beta[0]),
        factor_a_col: float(beta[1]),
        factor_b_col: float(beta[2]),
        f"{factor_a_col}:{factor_b_col}": float(beta[3]),
    }
    anova_df.attrs['r_squared'] = float(r_squared)
    anova_df.attrs['n_obs'] = int(df.shape[0])

    return anova_df, posthoc_df


def _resolve_ctrl_cask_groups(available_groups: list[str]) -> tuple[str, str]:
    """Resolve control and cask group names from aliases."""
    ctrl_group = _find_group_by_alias(available_groups, {'ctrl', 'control'})
    cask_group = _find_group_by_alias(available_groups, {'cask'})
    if ctrl_group is None or cask_group is None:
        raise KeyError("Could not resolve both control and cask groups from available data.")
    return ctrl_group, cask_group


def run_wsls_group_strategy_two_way_anova(
    win_stay_by_group: dict[str, list[float]],
    lose_shift_by_group: dict[str, list[float]],
    *,
    correction_method: str = 'holm',
) -> dict[str, pd.DataFrame | dict[str, str]]:
    """Two-way ANOVA for overall WS/LS with factors: group x strategy.

    Includes interaction testing and posthoc pairwise comparisons with
    multiple-comparison correction.
    """
    available_groups = sorted(set(win_stay_by_group.keys()) | set(lose_shift_by_group.keys()))
    ctrl_group, cask_group = _resolve_ctrl_cask_groups(available_groups)

    rows: list[dict[str, str | float]] = []
    for group in [ctrl_group, cask_group]:
        ws_vals = _clean_numeric_values(win_stay_by_group.get(group, []))
        ls_vals = _clean_numeric_values(lose_shift_by_group.get(group, []))
        for val in ws_vals:
            rows.append({'value': float(val), 'group': group, 'strategy': 'win-stay'})
        for val in ls_vals:
            rows.append({'value': float(val), 'group': group, 'strategy': 'lose-shift'})

    data_df = pd.DataFrame(rows, columns=['value', 'group', 'strategy'])
    anova_df, posthoc_df = _run_two_way_anova_with_posthoc(
        data_df,
        value_col='value',
        factor_a_col='group',
        factor_b_col='strategy',
        factor_a_order=[ctrl_group, cask_group],
        factor_b_order=['win-stay', 'lose-shift'],
        correction_method=correction_method,
    )

    print("[WSLS][Two-way ANOVA] Overall strategy model: value ~ group * strategy")
    print(f"Included groups: control={ctrl_group}, cask={cask_group}")
    print(anova_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"[WSLS][Posthoc] Corrected pairwise comparisons ({correction_method})")
    print(posthoc_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    return {
        'anova': anova_df,
        'posthoc': posthoc_df,
        'groups': {'control': ctrl_group, 'cask': cask_group},
    }


def run_wsls_group_window_two_way_anova(
    wsls_window_metrics: dict[str, dict[str, list[float]]],
    *,
    metric: str,
    pellet_window: int = 10,
    correction_method: str = 'holm',
) -> dict[str, pd.DataFrame | dict[str, str] | str]:
    """Two-way ANOVA for one metric with factors: group x time window.

    Args:
        wsls_window_metrics: Output dictionary from `wsls_pellet_window_by_group`
            or `wsls_pellet_window_from_session_analyses`.
        metric: Either 'win_stay' or 'lose_shift'.
        pellet_window: Number of pellets represented by the first/last windows.
        correction_method: Multiple-comparison correction for posthoc tests.
    """
    metric = str(metric)
    if metric not in {'win_stay', 'lose_shift'}:
        raise ValueError("metric must be 'win_stay' or 'lose_shift'.")

    first_key = f'{metric}_first'
    last_key = f'{metric}_last'
    if first_key not in wsls_window_metrics or last_key not in wsls_window_metrics:
        raise KeyError(f"Missing metric keys for group-window ANOVA: {first_key}, {last_key}")

    available_groups = sorted(set(wsls_window_metrics[first_key].keys()) | set(wsls_window_metrics[last_key].keys()))
    ctrl_group, cask_group = _resolve_ctrl_cask_groups(available_groups)

    rows: list[dict[str, str | float]] = []
    first_label = f'first {pellet_window}'
    last_label = f'last {pellet_window}'
    for group in [ctrl_group, cask_group]:
        first_vals = _clean_numeric_values(wsls_window_metrics[first_key].get(group, []))
        last_vals = _clean_numeric_values(wsls_window_metrics[last_key].get(group, []))
        for val in first_vals:
            rows.append({'value': float(val), 'group': group, 'window': first_label})
        for val in last_vals:
            rows.append({'value': float(val), 'group': group, 'window': last_label})

    data_df = pd.DataFrame(rows, columns=['value', 'group', 'window'])
    anova_df, posthoc_df = _run_two_way_anova_with_posthoc(
        data_df,
        value_col='value',
        factor_a_col='group',
        factor_b_col='window',
        factor_a_order=[ctrl_group, cask_group],
        factor_b_order=[first_label, last_label],
        correction_method=correction_method,
    )

    metric_label = metric.replace('_', '-')
    print(f"[WSLS][Two-way ANOVA] {metric_label} model: value ~ group * window")
    print(f"Included groups: control={ctrl_group}, cask={cask_group}")
    print(anova_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"[WSLS][Posthoc] Corrected pairwise comparisons ({correction_method})")
    print(posthoc_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    return {
        'metric': metric,
        'anova': anova_df,
        'posthoc': posthoc_df,
        'groups': {'control': ctrl_group, 'cask': cask_group},
    }


def run_wsls_group_window_two_way_anovas(
    wsls_window_metrics: dict[str, dict[str, list[float]]],
    *,
    pellet_window: int = 10,
    correction_method: str = 'holm',
) -> dict[str, dict[str, pd.DataFrame | dict[str, str] | str]]:
    """Run group x window two-way ANOVAs for both WSLS metrics.

    This produces two analyses:
      1) win-stay: group x window
      2) lose-shift: group x window
    """
    win_result = run_wsls_group_window_two_way_anova(
        wsls_window_metrics,
        metric='win_stay',
        pellet_window=pellet_window,
        correction_method=correction_method,
    )
    lose_result = run_wsls_group_window_two_way_anova(
        wsls_window_metrics,
        metric='lose_shift',
        pellet_window=pellet_window,
        correction_method=correction_method,
    )
    return {
        'win_stay': win_result,
        'lose_shift': lose_result,
    }


REV_CORRELATION_FEATURE_COLUMNS = [
    'Win_Stay',
    'Learning_Result',
    'Overall_Accuracy',
    'Lose_Shift',
    'First_Good_Meal_Time',
    'First_Good_Meal_Ratio',
]


def build_reversal_feature_table(feature_rows: list[dict]) -> pd.DataFrame:
    """Build a standardised per-mouse feature table for correlation analysis."""
    if not feature_rows:
        raise ValueError("feature_rows is empty; cannot build correlation table.")

    feature_df = pd.DataFrame(feature_rows).copy()
    for metadata_col in ['Group', 'Mouse_ID', 'Session_ID']:
        if metadata_col not in feature_df.columns:
            feature_df[metadata_col] = 'unknown'

    for feature_col in REV_CORRELATION_FEATURE_COLUMNS:
        if feature_col not in feature_df.columns:
            feature_df[feature_col] = np.nan
        feature_df[feature_col] = pd.to_numeric(feature_df[feature_col], errors='coerce')

    ordered_cols = ['Group', 'Mouse_ID', 'Session_ID'] + REV_CORRELATION_FEATURE_COLUMNS
    return feature_df[ordered_cols]


def plot_reversal_feature_correlation(
    feature_table: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    method: str = 'pearson',
    export_path: str | os.PathLike | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """Plot a feature-correlation heatmap using per-mouse reversal metrics."""
    if feature_columns is None:
        feature_columns = REV_CORRELATION_FEATURE_COLUMNS.copy()

    missing_columns = [col for col in feature_columns if col not in feature_table.columns]
    if missing_columns:
        raise KeyError(f"Missing correlation columns: {missing_columns}")

    numeric_df = feature_table[feature_columns].apply(pd.to_numeric, errors='coerce')
    valid_cols = [col for col in feature_columns if numeric_df[col].notna().sum() >= 2]
    if len(valid_cols) < 2:
        raise ValueError("At least two feature columns with >=2 values are required.")

    corr = numeric_df[valid_cols].corr(method=method)
    n_mice = int(numeric_df[valid_cols].dropna(how='all').shape[0])

    label_map = {
        'Win_Stay': 'Win-Stay',
        'Lose_Shift': 'Lose-Shift',
        'First_Good_Meal_Time': 'First Good Meal Time',
        'First_Good_Meal_Ratio': 'First Good Meal Ratio',
        'Learning_Result': 'Learning Result',
        'Overall_Accuracy': 'Overall Accuracy',
    }
    labels = [label_map.get(col, col) for col in valid_cols]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    heatmap = ax.imshow(corr.to_numpy(dtype=float), cmap='coolwarm', vmin=-1, vmax=1)

    tick_positions = np.arange(len(valid_cols))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_yticklabels(labels)

    for row_idx in range(len(valid_cols)):
        for col_idx in range(len(valid_cols)):
            val = corr.iat[row_idx, col_idx]
            if not np.isfinite(val):
                continue
            txt_color = 'white' if abs(val) >= 0.5 else 'black'
            ax.text(
                col_idx,
                row_idx,
                f"{val:.2f}",
                ha='center',
                va='center',
                fontsize=9,
                color=txt_color,
            )

    ax.set_title('Reversal Feature Correlation (Per Mouse)')
    n_handle = Line2D([], [], linestyle='None', marker=None, label=f'n_mice={n_mice}')
    ax.legend(handles=[n_handle], loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label('Pearson r')
    fig.tight_layout()

    if export_path:
        fig.savefig(export_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return corr


def accuracy(group: pd.DataFrame) -> float:
    """Calculate the percentage of poke events that match the active poke.

    Args:
        group (pd.DataFrame): Events from a block or subset of a block. Pellet
            rows are ignored when computing accuracy.

    Returns:
        float: Accuracy value expressed as a percentage.
    """
    group = group[group['Event'] != 'Pellet']
    total_events = len(group)
    matching_events = group[group['Event'] == group['Active_Poke']]
    matching_count = len(matching_events)

    if total_events == 0:
        return 0
    else:
        return (matching_count / total_events) * 100
    
    
def block_accuracy_by_proportion(blocks: list[pd.DataFrame], proportion: float) -> list[float]:
    """Measure accuracy within the leading portion of each block.

    Args:
        blocks (list[pd.DataFrame]): Behavioural blocks for one subject.
        proportion (float): Fraction of rows from the start of each block to
            include when computing accuracy.

    Returns:
        list[float]: Accuracy values for each block slice.
    """
    acc = []
    for block in blocks:
        size = int(len(block) * proportion)
        acc.append(accuracy(block[:size]))
    return acc


def learning_score(blocks: list[pd.DataFrame], block_prop: float = 0.5, action_prop: float = 0.8) -> float:
    """Summarise early-block accuracy for a subject.

    Args:
        blocks (list[pd.DataFrame]): Behavioural blocks ordered chronologically.
        block_prop (float): Fraction of initial blocks to evaluate.
        action_prop (float): Portion of each evaluated block used to calculate
            accuracy.

    Returns:
        float: Mean accuracy across the selected block slices.
    """
    cutoff = int(len(blocks)*block_prop)
    return np.mean(block_accuracy_by_proportion(blocks=blocks[:cutoff], proportion=action_prop))


def learning_result(blocks: list[pd.DataFrame], action_prop: float = 0.75) -> float:
    """Compute late-block accuracy to compare end-of-session performance.

    Args:
        blocks (list[pd.DataFrame]): Behavioural blocks ordered chronologically.
        action_prop (float): Fraction of each block to skip before measuring
            accuracy.

    Returns:
        float: Mean accuracy for the remaining portion of each block.
    """
    results = [accuracy(block[int(len(block)*action_prop):]) for block in blocks]
    return np.mean(results)



def graph_learning_score(
    ctrl: list,
    exp: list,
    width: float = 0.4,
    group_names: list | None = None,
    proportion: float | None = None,
    export_path: str | os.PathLike | None = None,
    verbose: bool = True,
):
    """Plot violin summaries of learning scores for two cohorts."""
    ctrl_mean, exp_mean = np.mean(ctrl), np.mean(exp)
    ctrl_se, exp_se = np.std(ctrl) / np.sqrt(len(ctrl)), np.std(exp) / np.sqrt(len(exp))

    if group_names is None or len(group_names) < 2:
        group_names = ['Control', 'Experiment']
    ctrl_name, exp_name = group_names

    if verbose:
        print(f'{ctrl_name} Size: {len(ctrl)}   Avg: {ctrl_mean:.3f}   SE: {ctrl_se:.3f}')
        print(f'{exp_name}  Size: {len(exp)}   Avg: {exp_mean:.3f}   SE: {exp_se:.3f}')

    fig, ax = plt.subplots(figsize=(7, 7))
    x_positions = [1, 2]
    data = [ctrl, exp]

    parts = ax.violinplot(
        data,
        positions=x_positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, violin in enumerate(parts['bodies']):
        face = 'lightblue' if i == 0 else 'yellow'
        violin.set_facecolor(face)
        violin.set_edgecolor('black')
        violin.set_alpha(0.8)

    ax.boxplot(
        data,
        positions=x_positions,
        widths=width * 0.5,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='white', edgecolor='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )

    jitter = width / 8
    x_ctrl = 1 + np.random.uniform(-jitter, jitter, size=len(ctrl))
    x_exp = 2 + np.random.uniform(-jitter, jitter, size=len(exp))
    ax.scatter(x_ctrl, ctrl, marker='o', zorder=3, color='#1405eb', alpha=0.8)
    ax.scatter(x_exp, exp, marker='o', zorder=3, color='#f28211', alpha=0.8)

    c_patch = mpatches.Patch(color='lightblue', alpha=0.8, label=f'{ctrl_name} (n={len(ctrl)})')
    e_patch = mpatches.Patch(color='yellow', alpha=0.8, label=f'{exp_name} (n={len(exp)})')
    ax.legend(handles=[c_patch, e_patch])

    ax.set_ylim(45, 65)
    ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylabel('Learning Score', fontsize=14)
    ax.set_title(f'Learning Score of {ctrl_name} vs {exp_name} ({proportion} data)', fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names)

    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()


def graph_learning_results(
    ctrl: list,
    exp: list,
    width: float = 0.4,
    group_names: list | None = None,
    proportion: float | None = None,
    export_path: str | os.PathLike | None = None,
    verbose: bool = True,
):
    """Visualise accuracy distributions (late-block performance) for two cohorts."""
    ctrl_mean, exp_mean = np.mean(ctrl), np.mean(exp)
    ctrl_se, exp_se = np.std(ctrl) / np.sqrt(len(ctrl)), np.std(exp) / np.sqrt(len(exp))

    if group_names is None or len(group_names) < 2:
        group_names = ['Control', 'Experiment']
    ctrl_name, exp_name = group_names

    if verbose:
        print(f'{ctrl_name} Size: {len(ctrl)}   Avg: {ctrl_mean:.3f}   SE: {ctrl_se:.3f}')
        print(f'{exp_name}  Size: {len(exp)}   Avg: {exp_mean:.3f}   SE: {exp_se:.3f}')

    fig, ax = plt.subplots(figsize=(7, 7))
    x_positions = [1, 2]
    data = [ctrl, exp]

    parts = ax.violinplot(
        data,
        positions=x_positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, violin in enumerate(parts['bodies']):
        face = 'lightblue' if i == 0 else 'yellow'
        violin.set_facecolor(face)
        violin.set_edgecolor('black')
        violin.set_alpha(0.8)

    ax.boxplot(
        data,
        positions=x_positions,
        widths=width * 0.5,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='white', edgecolor='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )

    jitter = width / 8
    x_ctrl = 1 + np.random.uniform(-jitter, jitter, size=len(ctrl))
    x_exp = 2 + np.random.uniform(-jitter, jitter, size=len(exp))
    ax.scatter(x_ctrl, ctrl, marker='o', zorder=3, color='#1405eb', alpha=0.8)
    ax.scatter(x_exp, exp, marker='o', zorder=3, color='#f28211', alpha=0.8)

    c_patch = mpatches.Patch(color='lightblue', alpha=0.8, label=f'{ctrl_name} (n={len(ctrl)})')
    e_patch = mpatches.Patch(color='yellow', alpha=0.8, label=f'{exp_name} (n={len(exp)})')
    ax.legend(handles=[c_patch, e_patch])

    ax.set_ylim(55, 85)
    ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylabel('Mean Accuracy (%)', fontsize=14)
    ax.set_title(f'Learning Result of {ctrl_name} vs {exp_name} (last {proportion} data)', fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names)

    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()
    

def plot_learning_score_trend(
    blocks_groups: list,
    group_labels: list | None = None,
    block_prop: float = 1.0,
    action_prop: float = 1.0,
    export_path: str | os.PathLike | None = None,
    n_bins: int = 19,
):
    """Plot learning-score curves for each group across increasing action proportions.

    Args:
        blocks_groups (list): Per-group collection of subject blocks. Each entry
            is a list where every element is the list of blocks for one subject.
        group_labels (list[str] | None): Optional names to use in the legend.
        block_prop (float): Fraction of blocks to include when computing each
            subject's learning score.
        action_prop (float): Reference proportion to highlight on the plot.
        export_path (str | os.PathLike | None): Optional path to save the figure.
        n_bins (int): Number of proportions (between 5% and 100%) to sample.

    Returns:
        None
    """
    if group_labels is None:
        group_labels = [f"Group {idx + 1}" for idx in range(len(blocks_groups))]
    if len(group_labels) != len(blocks_groups):
        raise ValueError("group_labels length must match blocks_groups length")

    proportions = np.linspace(0.05, 1.0, n_bins)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig, ax = plt.subplots(figsize=(10, 6))

    highlight_prop = action_prop if 0 < action_prop <= 1 else None
    highlight_points: list[tuple[float, float]] = []

    for group_idx, blocks_list in enumerate(blocks_groups):
        if not blocks_list:
            continue

        group_means = []
        group_sems = []

        for prop in proportions:
            mouse_scores = [
                learning_score(blocks, block_prop=block_prop, action_prop=prop)
                for blocks in blocks_list
            ]
            mean_score = float(np.mean(mouse_scores))
            sem_score = float(np.std(mouse_scores, ddof=0) / np.sqrt(len(mouse_scores)))
            group_means.append(mean_score)
            group_sems.append(sem_score)

        group_means_arr = np.asarray(group_means)
        group_sems_arr = np.asarray(group_sems)
        color = colors[group_idx % len(colors)]

        ax.plot(proportions * 100, group_means_arr, color=color, linewidth=2, label=group_labels[group_idx])
        ax.fill_between(
            proportions * 100,
            group_means_arr - group_sems_arr,
            group_means_arr + group_sems_arr,
            color=color,
            alpha=0.2,
        )

        if highlight_prop is not None:
            highlight_scores = [
                learning_score(blocks, block_prop=block_prop, action_prop=highlight_prop)
                for blocks in blocks_list
            ]
            if highlight_scores:
                mean_highlight = float(np.mean(highlight_scores))
                highlight_points.append((highlight_prop * 100, mean_highlight))

    if highlight_points:
        x_pos = highlight_points[0][0]
        ax.axvline(x=x_pos, color='#4c4c4c', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.scatter(
            [pt[0] for pt in highlight_points],
            [pt[1] for pt in highlight_points],
            color='#4c4c4c',
            marker='o',
            zorder=4,
            label=f'Action proportion {highlight_prop * 100:.0f}%',
        )

    ax.set_xlabel('Action Proportion (%)', fontsize=12)
    ax.set_ylabel('Learning Score (%)', fontsize=12)
    ax.set_title('Learning Score Trend Across Action Proportions', fontsize=14)
    ax.set_xlim(5, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if export_path:
        plt.savefig(export_path, bbox_inches='tight', dpi=300)
    plt.show()


def find_meal_pellet_counts(
    data: pd.DataFrame,
    time_threshold: float = 60,
    pellet_threshold: int = 2,
    method: str = 'paper',
) -> list[int]:
    """Return pellet counts for each detected meal within a block subset.

    Args:
        data (pd.DataFrame): Block data containing pellet events and timestamps.
        time_threshold (float): Maximum seconds between pellets to remain in the
            same meal.
        pellet_threshold (int): Minimum pellet count required for a meal.
        method (str): Meal detection method ('paper' or 'ipi').

    Returns:
        list[int]: Pellet counts for every qualifying meal.
    """
    meals, _ = find_meals_paper(
        data,
        time_threshold=time_threshold,
        pellet_threshold=pellet_threshold,
        accuracy_threshold=50.0,
        method=method,
    )

    counts = []
    for start, end in meals:
        subset = data[(data['Time'] >= start) & (data['Time'] <= end)]
        cnt = len(subset[subset['Event'] == 'Pellet'])
        counts.append(cnt)

    return counts


def pellet_ratio_for_block(
    block: pd.DataFrame,
    proportion: float,
    time_threshold: float = 60,
    pellet_threshold: int = 2,
    method: str = 'paper',
) -> float:
    """Measure how many pellets fall inside meals for the start of a block.

    Args:
        block (pd.DataFrame): Behavioural block with pellet events.
        proportion (float): Fraction of the block to analyse.
        time_threshold (float): Seconds allowed between pellets within a meal.
        pellet_threshold (int): Minimum pellet count to define a meal.
        method (str): Meal detection method.

    Returns:
        float: Ratio of pellets that belong to meals, or ``np.nan`` when no
        pellets occur in the slice.
    """
    n = int(len(block) * proportion)
    sub = block.iloc[:n]

    total_pellets = (sub['Event'] == 'Pellet').sum()
    if total_pellets == 0:
        return np.nan

    meal_counts = find_meal_pellet_counts(
        sub,
        time_threshold=time_threshold,
        pellet_threshold=pellet_threshold,
        method=method,
    )
    pellets_in_meals = sum(meal_counts)
    return pellets_in_meals / total_pellets


def plot_pellet_ratio_trend(
    blocks_groups: list[list[pd.DataFrame]],
    group_labels: list[str] | None = None,
    time_threshold: float = 60,
    pellet_threshold: int = 2,
    method: str = 'paper',
    export_path: str | os.PathLike | None = None,
):
    """Visualise pellet-in-meal ratios for each group with violin plots.

    Args:
        blocks_groups (list[list[pd.DataFrame]]): Per-group collection of
            subject block lists.
        group_labels (list[str] | None): Optional legend labels.
        time_threshold (float): Seconds allowed between pellet retrievals within
            a meal.
        pellet_threshold (int): Minimum pellets to count a meal.
        method (str): Meal detection method.
        export_path (str | os.PathLike | None): Optional destination to save the
            figure.

    Returns:
        None
    """
    if group_labels is None:
        group_labels = [f"Group {i+1}" for i in range(len(blocks_groups))]

    group_ratios = []

    for blocks_list in blocks_groups:
        mouse_ratios = []

        for sample_blocks in blocks_list:
            block_ratios = []
            for block_df in sample_blocks:
                ratio = pellet_ratio_for_block(
                    block_df,
                    proportion=1.0,
                    time_threshold=time_threshold,
                    pellet_threshold=pellet_threshold,
                    method=method,
                )
                if not np.isnan(ratio):
                    block_ratios.append(ratio)

            if block_ratios:
                mouse_avg_ratio = np.mean(block_ratios)
                mouse_ratios.append(mouse_avg_ratio)

        group_ratios.append(mouse_ratios)

    graph_group_stats(
        group_data=group_ratios,
        stats_name="Pellet-in-Meal Ratio",
        unit="ratio",
        group_names=group_labels,
        export_path=export_path,
    )


def block_retrieval_summary(blocks: list[pd.DataFrame], n_stds: int = 3) -> tuple[list, float, float]:
    """Calculate retrieval time statistics for each block with outlier removal.
    
    Args:
        blocks (list[pd.DataFrame]): List of behavioral blocks containing 'collect_time' column.
        n_stds (int, optional): Number of standard deviations for outlier removal. Defaults to 3.
    
    Returns:
        tuple: Contains:
            - block_means (list): Mean retrieval time for each block (in seconds).
            - pred (float): Predicted retrieval time at the end (linear extrapolation).
            - slope (float): Slope of the linear fit across blocks.
    """
    block_means = []
    for block in blocks:
        times = pd.to_numeric(block["collect_time"], errors="coerce")
        times = times[(times > 0) & times.notna()]
        if times.empty:
            continue
        mean = times.mean()
        std = times.std(ddof=0)
        if not np.isnan(std) and std > 0:
            cutoff = mean + n_stds * std
            times = times[times <= cutoff]
        if not times.empty:
            block_means.append(times.mean())

    if not block_means:
        return [], 0.0, 0.0

    x = np.arange(len(block_means))
    if len(block_means) > 1:
        slope, intercept = np.polyfit(x, block_means, 1)
        pred = slope * len(block_means) + intercept
    else:
        slope = 0.0
        pred = block_means[-1]
    return block_means, float(pred), float(slope)


def plot_retrieval_time_by_block(
    block_means: list,
    *,
    mouse_label: str,
    group_label: str | None = None,
    export_path: str | os.PathLike | None = None,
    show: bool = False
):
    """Plot mean retrieval time per block with linear trend line.
    
    Args:
        block_means (list): Mean retrieval times for each block.
        mouse_label (str): Mouse identifier for the title.
        group_label (str, optional): Group name for the title.
        export_path (str | os.PathLike, optional): Path to save the figure.
        show (bool, optional): Whether to display the figure. Defaults to False.
    """
    if not block_means:
        return

    block_indices = np.arange(len(block_means))
    slope, intercept = np.polyfit(block_indices, block_means, 1)
    fit_line = slope * block_indices + intercept

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(block_indices + 1, block_means, marker='*', color='#1f77b4', linewidth=2, label='Mean retrieval time')
    ax.plot(block_indices + 1, fit_line, color='#ff7f0e', linestyle='--', linewidth=2, label=f'Best fit (slope={slope:.2f})')

    ax.set_xlabel('Block index', fontsize=12)
    ax.set_ylabel('Mean retrieval time (seconds)', fontsize=12)
    title_parts = ['Retrieval time per block']
    if group_label:
        title_parts.append(f"Group {group_label}")
    title_parts.append(f"Mouse {mouse_label}")
    ax.set_title(' - '.join(title_parts), fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend()

    if export_path:
        fig.savefig(export_path, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cumulative_pellet_ratio_trend(
    blocks_by_group: dict,
    group_labels: list[str] | None = None,
    time_threshold: float = 60,
    pellet_threshold: int = 2,
    method: str = 'paper',
    n_bins: int = 19,
    export_path: str | os.PathLike | None = None,
):
    """Plot cumulative pellet-in-meal ratio trend across block proportions with subplots.

    This function creates a figure with two subplots:
    - Left subplot: Groups like 'cask' and 'ctrl'
    - Right subplot: Group like 'female'
    
    Each curve shows how the pellet-in-meal ratio evolves as more of each block
    is considered (from 5% to 100% of block events).

    Args:
        blocks_by_group (dict): Dictionary mapping group names to lists of subject
            blocks. Each value is a list where every element is the list of blocks
            for one subject, e.g., {'cask': [[blocks_m1], [blocks_m2], ...], ...}
        group_labels (list[str] | None): Optional explicit labels for the legend.
            If None, uses group names from blocks_by_group keys.
        time_threshold (float): Maximum seconds between pellets to remain in the
            same meal (default 60).
        pellet_threshold (int): Minimum pellet count required for a meal (default 2).
        method (str): Meal detection method ('paper' or 'ipi').
        n_bins (int): Number of proportions (between 5% and 100%) to sample.
        export_path (str | os.PathLike | None): Optional path to save the figure.

    Returns:
        None
    """
    if group_labels is None:
        group_labels = list(blocks_by_group.keys())
    
    proportions = np.linspace(0.05, 1.0, n_bins)
    
    # Define color palette for groups
    color_map = {
        'cask': '#1f77b4',    # blue
        'ctrl': '#ff7f0e',    # orange
        'female': '#2ca02c',  # green
    }
    default_colors = ['#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Separate groups into two categories
    # Left subplot: cask, ctrl (or any groups not 'female')
    # Right subplot: female
    left_groups = [g for g in group_labels if g.lower() != 'female']
    right_groups = [g for g in group_labels if g.lower() == 'female']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax_left, ax_right = axes
    
    def plot_group_trend(ax, groups, title_suffix):
        """Helper to plot trends for a set of groups on one axis."""
        color_idx = 0
        for group in groups:
            if group not in blocks_by_group:
                continue
            blocks_list = blocks_by_group[group]
            if not blocks_list:
                continue
            
            # Get color
            color = color_map.get(group.lower())
            if color is None:
                color = default_colors[color_idx % len(default_colors)]
                color_idx += 1
            
            group_means = []
            group_sems = []
            
            for prop in proportions:
                mouse_ratios = []
                for sample_blocks in blocks_list:
                    block_ratios = []
                    for block_df in sample_blocks:
                        ratio = pellet_ratio_for_block(
                            block_df,
                            proportion=prop,
                            time_threshold=time_threshold,
                            pellet_threshold=pellet_threshold,
                            method=method,
                        )
                        if not np.isnan(ratio):
                            block_ratios.append(ratio)
                    
                    if block_ratios:
                        mouse_avg_ratio = np.mean(block_ratios)
                        mouse_ratios.append(mouse_avg_ratio)
                
                if mouse_ratios:
                    mean_ratio = float(np.mean(mouse_ratios))
                    sem_ratio = float(np.std(mouse_ratios, ddof=0) / np.sqrt(len(mouse_ratios)))
                else:
                    mean_ratio = np.nan
                    sem_ratio = np.nan
                
                group_means.append(mean_ratio)
                group_sems.append(sem_ratio)
            
            group_means_arr = np.asarray(group_means)
            group_sems_arr = np.asarray(group_sems)
            
            # Plot line with error band
            n_subjects = len(blocks_list)
            ax.plot(
                proportions * 100, 
                group_means_arr, 
                color=color, 
                linewidth=2, 
                label=f'{group} (n={n_subjects})'
            )
            ax.fill_between(
                proportions * 100,
                group_means_arr - group_sems_arr,
                group_means_arr + group_sems_arr,
                color=color,
                alpha=0.2,
            )
        
        ax.set_xlabel('Block Proportion (%)', fontsize=12)
        ax.set_xlim(5, 100)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_title(f'Pellet-in-Meal Ratio Trend ({title_suffix})', fontsize=14)
    
    # Plot left subplot (cask, ctrl)
    if left_groups:
        plot_group_trend(ax_left, left_groups, 'cask & ctrl')
        ax_left.set_ylabel('Pellet-in-Meal Ratio', fontsize=12)
    else:
        ax_left.set_visible(False)
    
    # Plot right subplot (female)
    if right_groups:
        plot_group_trend(ax_right, right_groups, 'female')
    else:
        ax_right.set_visible(False)
    
    plt.tight_layout()
    
    if export_path:
        plt.savefig(export_path, bbox_inches='tight', dpi=300)
    plt.show()
