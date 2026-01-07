"""
This script analyzes behavioral data from FED3 experiments, focusing on transitions
between different states (e.g., left/right pokes) and learning performance. It
includes functions to split data into blocks, calculate transition statistics,
visualize learning trends, and assess learning scores.
"""
import os
from datetime import timedelta
from scripts.utils import graph_group_stats

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
