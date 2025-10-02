"""
This script analyzes behavioral data from FED3 experiments, focusing on transitions
between different states (e.g., left/right pokes) and learning performance. It
includes functions to split data into blocks, calculate transition statistics,
visualize learning trends, and assess learning scores.
"""
import os
import warnings
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from scripts.meals import find_meals_paper, find_first_good_meal, predict_meal_quality

colors = {'Left': 'red', 'Right': 'blue', 'Pellet': 'green'}


def split_data_to_blocks(data_dropped: pd.DataFrame, day=3) -> list:
    """Splits the behavioral data into blocks based on changes in the active poke.

    Args:
        data_dropped (pd.DataFrame): The input DataFrame with behavioral data.
        day (int, optional): The number of days of data to include. Defaults to 3.

    Returns:
        list: A list of DataFrames, where each DataFrame represents a block of
              continuous activity with the same active poke.
    """
    data_dropped = data_dropped[data_dropped['Time_passed'] < timedelta(days=day)]
    curr_poke = data_dropped['Active_Poke'][0]
    blocks = []
    start_idx = 0

    for key, val in data_dropped.iterrows():
        if val['Active_Poke'] != curr_poke:
            blocks.append(data_dropped.iloc[start_idx:key].reset_index(drop=True))  # add current block
            start_idx = key # update start_idx
            curr_poke = val['Active_Poke']   # update poke marker

    blocks.append(data_dropped.iloc[start_idx:].reset_index(drop=True))  # append the last block
    return blocks


def count_transitions(sub_frame: pd.DataFrame) -> dict:
    """Counts the occurrences of different types of transitions between pokes.

    Args:
        sub_frame (pd.DataFrame): A DataFrame representing a single block of behavior.

    Returns:
        dict: A dictionary containing counts for each transition type and successful pokes.
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
    """Counts the number of pellet events in a given data frame.

    Args:
        sub_frame (pd.DataFrame): The DataFrame to process.

    Returns:
        int: The number of pellet events.
    """
    pellet_count = 0
    
    for _, row in sub_frame.iterrows():
        event = row['Event']
        
        if 'Pellet' in event:
            pellet_count += 1
    
    return pellet_count


def remove_pellet(block: pd.DataFrame) -> pd.DataFrame:
    """Removes pellet events from a block of data.

    Args:
        block (pd.DataFrame): The input block.

    Returns:
        pd.DataFrame: The block with pellet events removed.
    """
    return block[block['Event'] != 'Pellet']


def get_transition_info(blocks: list, meal_config:list, reverse:bool) -> pd.DataFrame:
    """Calculates transition-related statistics for each block of data.

    Args:
        blocks (list): A list of DataFrames, each representing a block.
        meal_config (list): A list containing time and pellet thresholds for meal definition.
        reverse (bool): A flag indicating whether to reverse the active/inactive periods.

    Returns:
        pd.DataFrame: A DataFrame with detailed statistics for each block.
    """
    new_add = []
    inactives = find_inactive_blocks(blocks, reverse=reverse)

    for i, block in enumerate(blocks):
        no_pellet = remove_pellet(block)
        size = len(no_pellet)
        transitions = count_transitions(no_pellet)
        active_poke = block.iloc[0]['Active_Poke']

        times = block['Time'].tolist()
        block_time = round((times[-1] - times[0]).total_seconds() / 60, 2)
        meals,_ = find_meals_paper(block, meal_config[0], meal_config[1])
        time = round((meals[0][0] - times[0]).total_seconds() / 60, 2) if len(meals) > 0 else 'no meal'

        _, first_meal_time = find_first_good_meal(block, 60, 2, 'cnn')
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


def first_meal_stats(data_stats: pd.DataFrame, ignore_inactive=False):
    """Calculates statistics about the first meal in each block.

    Args:
        data_stats (pd.DataFrame): DataFrame with block statistics.
        ignore_inactive (bool, optional): If True, inactive blocks are excluded. Defaults to False.

    Returns:
        tuple: A tuple containing the average ratio of first good meal time to block time,
               average first meal time, and median first good meal time.
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


def find_inactive_blocks(blocks:list, reverse):
    """Identifies blocks that occur during the inactive (light) period.

    Args:
        blocks (list): A list of DataFrames, each representing a block.
        reverse (bool): If True, identifies blocks in the active (inactive) period instead.

    Returns:
        list: A list of indices of the inactive blocks.
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
    blocks: list,
    *,
    mouse_label: str,
    group_label: str | None = None,
    export_path: str | os.PathLike | None = None,
    show: bool = False,
    inactive_reverse: bool = False,
) -> None:
    """Plot transition statistics, success rates, and block context for one session."""

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
    if success_rate is not None:
        bars = ax.bar(block_idx, success_rate, color=bar_palette, alpha=0.6)

    first_good = data_stats.get('First_Good_Meal_Time', pd.Series([''] * len(block_idx)))
    block_time = data_stats.get('Block_Time', pd.Series([np.nan] * len(block_idx)))

    if bars is not None:
        for bar, meal_time, total_time in zip(bars, first_good, block_time):
            center_x = bar.get_x() + bar.get_width() / 2
            label_text = str(meal_time)
            total_text = f"{total_time}" if pd.notna(total_time) else ""
            ax.text(center_x, bar.get_height() + 2.4, label_text, ha='center', va='bottom', fontsize=10)
            if total_text:
                ax.text(center_x, bar.get_height() + 0.6, total_text, ha='center', va='bottom', fontsize=10, color='#555555')

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


def graph_tranition_stats(data_stats: pd.DataFrame, blocks: list, sheet: str, export_path=None):
    """Backward compatible wrapper for the legacy API."""
    warnings.warn(
        "graph_tranition_stats is deprecated; use plot_transition_stats with explicit labels instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    plot_transition_stats(
        data_stats,
        blocks,
        mouse_label=sheet,
        group_label=None,
        export_path=export_path,
        show=False,
    )
    return None


def accuracy(group: pd.DataFrame):
    """Calculates the percentage of correct pokes in a given interval.

    Args:
        group (pd.DataFrame): DataFrame for a specific interval.

    Returns:
        float: The accuracy percentage (0-100).
    """ 
    group = group[group['Event'] != 'Pellet']
    total_events = len(group)
    matching_events = group[group['Event'] == group['Active_Poke']]
    matching_count = len(matching_events)

    if total_events == 0:
        return 0
    else:
        return (matching_count / total_events) * 100
    
    
def block_accuracy_by_proportion(blocks: list, proportion: float):
    """Calculates the accuracy for a specified proportion of each block.

    Args:
        blocks (list): A list of block DataFrames.
        proportion (float): The proportion of each block to consider for accuracy calculation.

    Returns:
        list: A list of accuracy values for each block.
    """
    acc = []
    for block in blocks:
        size = int(len(block) * proportion)
        acc.append(accuracy(block[:size]))
    return acc


def learning_score(blocks: list, block_prop=0.5, action_prop=0.8) -> float:
    """Calculates a learning score based on accuracy in a proportion of blocks and actions.

    Args:
        blocks (list): A list of block DataFrames.
        block_prop (float, optional): The proportion of initial blocks to use. Defaults to 0.5.
        action_prop (float, optional): The proportion of initial actions within each block to use. Defaults to 0.8.

    Returns:
        float: The calculated learning score.
    """
    cutoff = int(len(blocks)*block_prop)
    return np.mean(block_accuracy_by_proportion(blocks=blocks[:cutoff], proportion=action_prop))


def learning_result(blocks, action_prop=0.75) -> float:
    """Calculates the final performance accuracy on the later part of each block.

    Args:
        blocks (list): A list of block DataFrames.
        action_prop (float, optional): The rest of actions from action_prop to the end of each block. Defaults to 0.75.

    Returns:
        float: The mean accuracy over the specified final portions of the blocks.
    """
    results = [accuracy(block[int(len(block)*action_prop):]) for block in blocks]
    return np.mean(results)



def graph_learning_score(ctrl: list,
                         exp: list,
                         width=0.4,
                         group_names=None,
                         proportion=None,
                         export_path=None,
                         verbose=True):
    """Graphs the learning scores of two groups using a violin plot with an inset box plot.

    Args:
        ctrl (list): A list of learning scores for the control group.
        exp (list): A list of learning scores for the experimental group.
        width (float, optional): The width of the violin plots. Defaults to 0.4.
        group_names (list, optional): Names for the groups. Defaults to ['Control', 'Experiment'].
        proportion (float, optional): The proportion of data used, for labeling. Defaults to None.
        export_path (str, optional): Path to save the plot. Defaults to None.
        verbose (bool, optional): If True, prints summary statistics. Defaults to True.
    """
    # summary stats
    ctrl_mean, exp_mean = np.mean(ctrl), np.mean(exp)
    ctrl_se,   exp_se   = np.std(ctrl)/np.sqrt(len(ctrl)), np.std(exp)/np.sqrt(len(exp))

    if group_names is None or len(group_names) < 2:
        group_names = ['Control', 'Experiment']
    ctrl_name, exp_name = group_names

    if verbose:
        print(f'{ctrl_name} Size: {len(ctrl)}   Avg: {ctrl_mean:.3f}   SE: {ctrl_se:.3f}')
        print(f'{exp_name}  Size: {len(exp)}   Avg: {exp_mean:.3f}   SE: {exp_se:.3f}')

    # plot setup
    fig, ax = plt.subplots(figsize=(7,7))
    x_positions = [1, 2]
    data = [ctrl, exp]

    # violin
    parts = ax.violinplot(data,
                          positions=x_positions,
                          widths=width,
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)
    for i, violin in enumerate(parts['bodies']):
        face = 'lightblue' if i == 0 else 'yellow'
        violin.set_facecolor(face)
        violin.set_edgecolor('black')
        violin.set_alpha(0.8)

    # inset boxplot
    ax.boxplot(data,
               positions=x_positions,
               widths=width*0.5,
               showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))

    # jittered scatters
    jitter = width/8
    x_ctrl = 1 + np.random.uniform(-jitter, jitter, size=len(ctrl))
    x_exp  = 2 + np.random.uniform(-jitter, jitter, size=len(exp))
    ax.scatter(x_ctrl, ctrl, marker='o', zorder=3, color='#1405eb', alpha=0.8)
    ax.scatter(x_exp,  exp,  marker='o', zorder=3, color='#f28211', alpha=0.8)

    # legend
    c_patch = mpatches.Patch(color='lightblue', alpha=0.8, label=f'{ctrl_name} (n={len(ctrl)})')
    e_patch = mpatches.Patch(color='yellow',    alpha=0.8, label=f'{exp_name} (n={len(exp)})')
    ax.legend(handles=[c_patch, e_patch])

    # labels & title
    ax.set_ylim(45, 65)
    ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylabel('Learning Score', fontsize=14)
    ax.set_title(f'Learning Score of {ctrl_name} vs {exp_name} ({proportion} data)', fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names)

    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()


def graph_learning_results(ctrl: list,
                           exp: list,
                           width=0.4,
                           group_names=None,
                           proportion=None,
                           export_path=None,
                           verbose=True):
    """Graphs the learning results (accuracy) of two groups using a violin and box plot.

    Args:
        ctrl (list): List of accuracy scores for the control group.
        exp (list): List of accuracy scores for the experimental group.
        width (float, optional): Width of the violin plots. Defaults to 0.4.
        group_names (list, optional): Names for the groups. Defaults to None.
        proportion (float, optional): Proportion of data used for labeling. Defaults to None.
        export_path (str, optional): Path to save the plot. Defaults to None.
        verbose (bool, optional): If True, prints summary statistics. Defaults to True.
    """
    ctrl_mean, exp_mean = np.mean(ctrl), np.mean(exp)
    ctrl_se,   exp_se   = np.std(ctrl)/np.sqrt(len(ctrl)), np.std(exp)/np.sqrt(len(exp))

    if group_names is None or len(group_names) < 2:
        group_names = ['Control', 'Experiment']
    ctrl_name, exp_name = group_names

    if verbose:
        print(f'{ctrl_name} Size: {len(ctrl)}   Avg: {ctrl_mean:.3f}   SE: {ctrl_se:.3f}')
        print(f'{exp_name}  Size: {len(exp)}   Avg: {exp_mean:.3f}   SE: {exp_se:.3f}')

    fig, ax = plt.subplots(figsize=(7,7))
    x_positions = [1,2]
    data = [ctrl, exp]

    # violin
    parts = ax.violinplot(data,
                          positions=x_positions,
                          widths=width,
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)
    for i, violin in enumerate(parts['bodies']):
        face = 'lightblue' if i == 0 else 'yellow'
        violin.set_facecolor(face)
        violin.set_edgecolor('black')
        violin.set_alpha(0.8)

    # inset boxplot
    ax.boxplot(data,
               positions=x_positions,
               widths=width*0.5,
               showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))

    # scatter
    jitter = width/8
    x_ctrl = 1 + np.random.uniform(-jitter, jitter, size=len(ctrl))
    x_exp  = 2 + np.random.uniform(-jitter, jitter, size=len(exp))
    ax.scatter(x_ctrl, ctrl, marker='o', zorder=3, color='#1405eb', alpha=0.8)
    ax.scatter(x_exp,  exp,  marker='o', zorder=3, color='#f28211', alpha=0.8)

    # legend
    c_patch = mpatches.Patch(color='lightblue', alpha=0.8, label=f'{ctrl_name} (n={len(ctrl)})')
    e_patch = mpatches.Patch(color='yellow',    alpha=0.8, label=f'{exp_name} (n={len(exp)})')
    ax.legend(handles=[c_patch, e_patch])

    # labels & title
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
    group_labels: list = None,
    block_prop: float = 1.0,
    action_prop: float = 1.0,
    export_path: str = None
):
    """Plots learning scores using both violin plots and trend line plots.

    Creates two plots:
    1. Violin plot of final learning scores (at specified action_prop)
    2. Line plot showing learning score trend across different action proportions with SEM

    Args:
        blocks_groups (list): A list of groups, where each group is a list of block lists (one per subject).
        group_labels (list, optional): Names for each group. Defaults to None.
        block_prop (float, optional): The proportion of blocks to use. Defaults to 1.0.
        action_prop (float, optional): The action proportion to evaluate for violin plot. Defaults to 1.0 (100%).
        export_path (str, optional): Path to save the figure. Defaults to None.
    """

    # Default group labels
    if group_labels is None:
        group_labels = [f"Group {i+1}" for i in range(len(blocks_groups))]

    # Create trend line plot
    trend_export = export_path.replace('_overall', '_trend')
    _plot_learning_score_trend_lines(
        blocks_groups=blocks_groups,
        group_labels=group_labels,
        block_prop=block_prop,
        export_path=trend_export
    )


def _plot_learning_score_trend_lines(
    blocks_groups: list,
    group_labels: list,
    block_prop: float = 1.0,
    export_path: str = None,
    n_bins: int = 19
):
    """Helper function to create trend line plots for learning scores.
    
    Args:
        blocks_groups (list): A list of groups, where each group is a list of block lists (one per subject).
        group_labels (list): Names for each group.
        block_prop (float): The proportion of blocks to use.
        export_path (str, optional): Path to save the plot.
        n_bins (int): Number of bins for the trend analysis.
    """
    # Define proportion bins from 0.1 to 1.0
    proportions = np.linspace(0.05, 1.0, n_bins)
    
    # Colors for different groups
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for group_idx, blocks_list in enumerate(blocks_groups):
        group_means = []
        group_sems = []

        for prop in proportions:
            # Calculate learning scores for all mice in this group at this proportion
            mouse_scores = []
            for blocks in blocks_list:
                score = learning_score(blocks, block_prop=block_prop, action_prop=prop)
                mouse_scores.append(score)
            
            # Calculate mean and SEM
            mean_score = np.mean(mouse_scores)
            sem_score = np.std(mouse_scores) / np.sqrt(len(mouse_scores))
            
            group_means.append(mean_score)
            group_sems.append(sem_score)
        
        # Convert to numpy arrays
        group_means = np.array(group_means)
        group_sems = np.array(group_sems)
        
        # Plot line and shaded error region
        color = colors[group_idx % len(colors)]
        ax.plot(proportions * 100, group_means, 
                color=color, linewidth=2,
                label=group_labels[group_idx])
        ax.fill_between(proportions * 100, 
                       group_means - group_sems, 
                       group_means + group_sems,
                       color=color, alpha=0.2)
    
    # Formatting
    ax.set_xlabel('Action Proportion (%)', fontsize=12)
    ax.set_ylabel('Learning Score (%)', fontsize=12)
    ax.set_title('Learning Score Trend Across Action Proportions', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 100)
    
    if export_path:
        plt.savefig(export_path, bbox_inches='tight', dpi=300)
    plt.show()


def find_meal_pellet_counts(
    data: pd.DataFrame,
    time_threshold: float = 60,
    pellet_threshold: int = 2
) -> list[int]:
    """Scans for pellet events and returns a list of pellet counts for each meal.

    A meal is defined as a series of pellet events where each event is within
    `time_threshold` seconds of the start of the meal. Only meals with at least
    `pellet_threshold` pellets are counted.

    Args:
        data (pd.DataFrame): The input DataFrame containing event data.
        time_threshold (float, optional): The maximum time between pellets in a meal. Defaults to 60.
        pellet_threshold (int, optional): The minimum number of pellets to constitute a meal. Defaults to 2.

    Returns:
        list[int]: A list of pellet counts for each identified meal.
    """
    # narrow to pellet rows & compute retrieval timestamps
    df = data.loc[data['Event'] == 'Pellet'] .copy()
    df['retrieval_timestamp'] = (
        df['Time'] + pd.to_timedelta(df['collect_time'], unit='m')
    )

    meals = []
    pellet_cnt = 0
    meal_start_time = None

    for _, row in df.iterrows():
        t = row['retrieval_timestamp']
        if meal_start_time is None:
            # start new meal
            meal_start_time = t
            pellet_cnt = 1
        else:
            # still in same meal?
            if (t - meal_start_time).total_seconds() <= time_threshold:
                pellet_cnt += 1
            else:
                # close out old meal
                if pellet_cnt >= pellet_threshold:
                    meals.append(pellet_cnt)
                # start a fresh meal
                meal_start_time = t
                pellet_cnt = 1

    # final meal
    if pellet_cnt >= pellet_threshold:
        meals.append(pellet_cnt)

    return meals


def pellet_ratio_for_block(
    block: pd.DataFrame,
    proportion: float,
    time_threshold: float = 60,
    pellet_threshold: int = 2
) -> float:
    """Calculates the ratio of pellets within meals to total pellets for a proportion of a block.

    Args:
        block (pd.DataFrame): The DataFrame for a single block.
        proportion (float): The proportion of the block to analyze.
        time_threshold (float, optional): Time threshold for defining a meal. Defaults to 60.
        pellet_threshold (int, optional): Pellet threshold for defining a meal. Defaults to 2.

    Returns:
        float: The ratio of pellets in meals to total pellets. Returns np.nan if no pellets.
    """
    n = int(len(block) * proportion)
    sub = block.iloc[:n]

    total_pellets = (sub['Event'] == 'Pellet').sum()
    if total_pellets == 0:
        return np.nan

    meal_counts = find_meal_pellet_counts(
        sub,
        time_threshold=time_threshold,
        pellet_threshold=pellet_threshold
    )
    pellets_in_meals = sum(meal_counts)
    return pellets_in_meals / total_pellets


def plot_pellet_ratio_trend(
    blocks_groups: list[list[pd.DataFrame]],
    group_labels: list[str] = None,
    time_threshold: float = 60,
    pellet_threshold: int = 2,
    export_path: str = None
):
    """Plots pellet-in-meal ratios using both violin plots and trend line plots.

    Creates two plots:
    1. Violin plot of final pellet ratios (at 100% proportion)
    2. Line plot showing pellet ratio trend across different proportions with SEM

    Args:
        blocks_groups (list[list[pd.DataFrame]]): A list of groups, each containing lists of blocks for subjects.
        group_labels (list[str], optional): Names for each group. Defaults to None.
        time_threshold (float, optional): Time threshold for meal definition. Defaults to 60.
        pellet_threshold (int, optional): Pellet threshold for meal definition. Defaults to 2.
        export_path (str, optional): Path to save the plot. Defaults to None.
    """
    from scripts.accuracy import graph_group_stats
    
    # Default group labels
    if group_labels is None:
        group_labels = [f"Group {i+1}" for i in range(len(blocks_groups))]
    
    # Calculate average pellet ratio at 100% proportion for each mouse in each group (for violin plot)
    group_ratios = []
    
    for blocks_list in blocks_groups:
        mouse_ratios = []
        
        for sample_blocks in blocks_list:
            # Calculate pellet ratio for each block at 100% proportion
            block_ratios = []
            for block_df in sample_blocks:
                ratio = pellet_ratio_for_block(
                    block_df, 
                    proportion=1.0,  # 100% proportion
                    time_threshold=time_threshold,
                    pellet_threshold=pellet_threshold
                )
                if not np.isnan(ratio):
                    block_ratios.append(ratio)
            
            # Average across all blocks for this mouse
            if block_ratios:
                mouse_avg_ratio = np.mean(block_ratios)
                mouse_ratios.append(mouse_avg_ratio)
        
        group_ratios.append(mouse_ratios)
    
    # Create violin plot (existing functionality)
    violin_export = export_path
    graph_group_stats(
        group_data=group_ratios,
        stats_name="Pellet-in-Meal Ratio",
        unit="ratio",
        group_names=group_labels,
        export_path=violin_export
    )

    if len(group_ratios) == 2:
        from scripts.intervals import perform_T_test
        perform_T_test(group_ratios[0], group_ratios[1])
    
    # Create trend line plot
    trend_export = export_path.replace('_overall', '_trend')
    _plot_pellet_ratio_trend_lines(
        blocks_groups=blocks_groups,
        group_labels=group_labels,
        time_threshold=time_threshold,
        pellet_threshold=pellet_threshold,
        export_path=trend_export
    )


def _plot_pellet_ratio_trend_lines(
    blocks_groups: list[list[pd.DataFrame]],
    group_labels: list[str],
    time_threshold: float = 60,
    pellet_threshold: int = 2,
    export_path: str = None,
    n_bins: int = 20
):
    """Helper function to create trend line plots for pellet-in-meal ratios.
    
    Args:
        blocks_groups (list[list[pd.DataFrame]]): A list of groups, each containing lists of blocks for subjects.
        group_labels (list[str]): Names for each group.
        time_threshold (float): Time threshold for meal definition.
        pellet_threshold (int): Pellet threshold for meal definition.
        export_path (str, optional): Path to save the plot.
        n_bins (int): Number of bins for the trend analysis.
    """
    # Define proportion bins from 0.1 to 1.0
    proportions = np.linspace(0.0, 1.0, n_bins)
    
    # Colors for different groups
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for group_idx, blocks_list in enumerate(blocks_groups):
        group_means = []
        group_sems = []
        
        for prop in proportions:
            # Calculate pellet ratios for all mice in this group at this proportion
            mouse_ratios = []
            for sample_blocks in blocks_list:
                # Calculate average pellet ratio across all blocks for this mouse at this proportion
                block_ratios = []
                for block_df in sample_blocks:
                    ratio = pellet_ratio_for_block(
                        block_df,
                        proportion=prop,
                        time_threshold=time_threshold,
                        pellet_threshold=pellet_threshold
                    )
                    if not np.isnan(ratio):
                        block_ratios.append(ratio)
                
                # Average across blocks for this mouse
                if block_ratios:
                    mouse_avg_ratio = np.mean(block_ratios)
                    mouse_ratios.append(mouse_avg_ratio)
            
            # Calculate mean and SEM across mice
            if mouse_ratios:
                mean_ratio = np.mean(mouse_ratios)
                sem_ratio = np.std(mouse_ratios) / np.sqrt(len(mouse_ratios))
            else:
                mean_ratio = np.nan
                sem_ratio = np.nan
            
            group_means.append(mean_ratio)
            group_sems.append(sem_ratio)
        
        # Convert to numpy arrays and filter out NaN values
        group_means = np.array(group_means)
        group_sems = np.array(group_sems)
        
        # Only plot if we have valid data
        valid_mask = ~np.isnan(group_means)
        if np.any(valid_mask):
            valid_proportions = proportions[valid_mask]
            valid_means = group_means[valid_mask]
            valid_sems = group_sems[valid_mask]
            
            # Plot line and shaded error region
            color = colors[group_idx % len(colors)]
            ax.plot(valid_proportions * 100, valid_means, 
                    color=color, linewidth=2, marker='o', markersize=4,
                    label=group_labels[group_idx])
            ax.fill_between(valid_proportions * 100, 
                           valid_means - valid_sems, 
                           valid_means + valid_sems,
                           color=color, alpha=0.2)
    
    # Formatting
    ax.set_xlabel('Block Proportion (%)', fontsize=12)
    ax.set_ylabel('Pellet-in-Meal Ratio', fontsize=12)
    ax.set_title('Pellet-in-Meal Ratio Trend Across Block Proportions', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    
    if export_path:
        plt.savefig(export_path, bbox_inches='tight', dpi=300)
    plt.show()
