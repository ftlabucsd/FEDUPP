"""
This script analyzes behavioral data from FED3 experiments, focusing on transitions
between different states (e.g., left/right pokes) and learning performance. It
includes functions to split data into blocks, calculate transition statistics,
visualize learning trends, and assess learning scores.
"""
import os
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from meals import find_meals_paper, find_first_good_meal, pellet_flip
from preprocessing import get_bhv_num

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
        reverse (bool): If True, identifies blocks in the active (dark) period instead.

    Returns:
        list: A list of indices of the inactive blocks.
    """
    night_blocks = []
    block_start_index = 1

    for block_df in blocks:
        if not block_df.empty and 'Time' in block_df:
            times = pd.to_datetime(block_df['Time']).tolist()
            cnt = [1 if time.hour >= 19 or time.hour < 7 else 0 for time in times]
            if sum(cnt) > len(cnt) // 2:
                night_blocks.append(block_start_index)
        block_start_index += 1

    if reverse:
        night_blocks = [each for each in range(1, len(blocks)+1) if each not in night_blocks]
    return night_blocks


def graph_tranition_stats(data_stats: pd.DataFrame, blocks: list, sheet: str, export_path=None):
    """Graphs transition statistics, success rates, and active pokes for each block.

    Args:
        data_stats (pd.DataFrame): DataFrame with transition statistics.
        blocks (list): A list of block DataFrames.
        sheet (str): The name of the data sheet, used for titling the plot.
        export_path (str, optional): Path to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(22, 12))

    l1 = ax.plot(data_stats['Block_Index'], data_stats['Left_to_Left'],
            marker='o', label='Left-Left', color='black')
    l2 = ax.plot(data_stats['Block_Index'], data_stats['Left_to_Right'],
            marker='*', label='Left-Right', color='green')
    l3 = ax.plot(data_stats['Block_Index'], data_stats['Right_to_Right'],
            marker='s', label='Right-Right', color='orange')
    l4 = ax.plot(data_stats['Block_Index'], data_stats['Right_to_Left'],
            marker='X', label='Right-Left', color='purple')
     
    bars1 = ax.bar(data_stats['Block_Index'][::2], data_stats['Success_Rate'][::2],
                color='pink' if data_stats['Active_Poke'][0] == 'Left' else 'lightblue', alpha=0.7)

    bars2 = ax.bar(data_stats['Block_Index'][1::2], data_stats['Success_Rate'][1::2],
                color='lightblue' if data_stats['Active_Poke'][1] == 'Right' else 'pink', alpha=0.7)

    labels = data_stats['First_Good_Meal_Time']
    total_times = data_stats['Block_Time']
    for bar, label, total in zip(bars1, labels[::2], total_times[::2]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.75,  # Add small offset (0.01) to bar height
                label, ha='center', va='bottom', fontsize=12)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,  # Add small offset (0.01) to bar height
                total, ha='center', va='bottom', fontsize=12)
            
    for bar, label, total in zip(bars2, labels[1::2], total_times[1::2]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.75,  # Add small offset (0.01) to bar height
                label, ha='center', va='bottom', fontsize=12)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,  # Add small offset (0.01) to bar height
                total, ha='center', va='bottom', fontsize=12)
    
    plt.annotate('First Accurate Meal Time (min) \n Block Length', 
            xy=(bars1[0].get_x()+0.4, bars1[0].get_height() + 4.2), 
            xytext=(-90, 20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', lw=2),
            fontsize=16,
            color='blue')

    ax.set_xlabel('Blocks', fontsize=16)
    ax.set_ylabel('Percentage(%)', color='black', fontsize=16)

    info = get_bhv_num(sheet)
    night_blocks = find_inactive_blocks(blocks, reverse=False)

    for block_index in night_blocks:
        ax.axvspan(block_index - 0.5, block_index + 0.5, facecolor='gray', alpha=0.4)

    left_patch = mpatches.Patch(color='pink', alpha=0.5, label='Left Active')
    right_patch = mpatches.Patch(color='lightblue', alpha=0.5, label='Right Active')
    night_patch = mpatches.Patch(color='gray', alpha=0.5, label='Inactive Period')

    lines = [l1[0], l2[0], l3[0], l4[0], bars1[0], bars2[0], left_patch, right_patch, night_patch]
    labels = [line.get_label() for line in lines]  # Get the labels of the line
    legend = ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.13, 1), borderaxespad=0)
    legend.set_title('Legend')
    legend.get_title().set_fontsize('16')
    for text in legend.get_texts():
        text.set_fontsize('14')
    
    if len(info) == 2:
        plt.title(f'Probability of Transitions in Poke Choosing and Accuracy in Group {info[0]} Mouse {info[1]}', fontsize=24)
    else:
        plt.title(f'Probability of Transitions in Poke Choosing and Accuracy Rates of Mouse {info[0]}', fontsize=24)

    plt.xticks(data_stats['Block_Index'])
    plt.yticks(range(0, 100, 20))
    fig.set_dpi(150)
    ax.grid(alpha=0.5, linestyle='--')
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
        return
    plt.show()
    
    
def graph_learning_trend(data_stats: pd.DataFrame, blocks: list, path: str, block_prop=0.6, action_prop=0.5):
    """Graphs the learning trend over a proportion of blocks.

    Args:
        data_stats (pd.DataFrame): DataFrame with block statistics.
        blocks (list): List of block DataFrames.
        path (str): Path to the original data file for metadata.
        block_prop (float, optional): Proportion of blocks to analyze. Defaults to 0.6.
        action_prop (float, optional): Proportion of actions within each block to analyze. Defaults to 0.5.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    cutoff = int(len(data_stats)*block_prop)
    data_stats = data_stats[:cutoff]
    cut_blocks = blocks[:cutoff]
    acc_in_block_by_prop = block_accuracy_by_proportion(cut_blocks, action_prop)

    ax.bar(data_stats['Block_Index'][::2], acc_in_block_by_prop[::2],
        label='Success Rate', color='pink' if data_stats['Active_Poke'][0] == 'Left' else 'lightblue', alpha=0.7)
    ax.bar(data_stats['Block_Index'][1::2], acc_in_block_by_prop[1::2],
        label='Success Rate', color='lightblue' if data_stats['Active_Poke'][1] == 'Right' else 'pink', alpha=0.7)

    ax.set_xlabel('Blocks', fontsize=16)
    ax.set_ylabel('Percentage(%)', color='black', fontsize=16)

    info = get_bhv_num(path)
    night_blocks = find_inactive_blocks(blocks, False)
    
    for block_index in night_blocks:
        ax.axvspan(block_index - 0.5, block_index + 0.5, facecolor='gray', alpha=0.4)

    left_patch = mpatches.Patch(color='pink', alpha=0.5, label='Left Active')
    right_patch = mpatches.Patch(color='lightblue', alpha=0.5, label='Right Active')
    night_patch = mpatches.Patch(color='gray', alpha=0.5, label='Inactive Period')

    leg_bg = plt.legend(handles=[left_patch, right_patch, night_patch], loc='upper right')

    leg_bg.set_title('Correct Rate')
    leg_bg.get_title().set_fontsize('17')
    leg_bg.get_texts()[0].set_fontsize('15')
    leg_bg.get_texts()[1].set_fontsize('15')
    leg_bg.get_texts()[2].set_fontsize('15')

    if len(info) == 2:
        plt.title(f'Accuracy by Switch for Group {info[0]} Mouse {info[1]}', fontsize=24)
    else:
        plt.title(f'Accuracy by Switch for Mouse {info[0]}', fontsize=24)

    plt.xticks(data_stats['Block_Index'])
    plt.yticks(range(0, 100, 20))
    fig.set_dpi(80)
    plt.grid(alpha=0.5, linestyle='--')
    plt.show()


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
    ax.set_ylim(20, 65)
    ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylabel('Learning Score', fontsize=14)
    ax.set_title(f'Learning Score of {ctrl_name} vs {exp_name} ({proportion} data)', fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names)

    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()


def graph_learning_score_single(data: list,
                                width=0.4,
                                group_name=None,
                                proportion=None,
                                export_path=None,
                                verbose=True):
    """Graphs the learning score of a single group using a violin plot with a box plot.

    Args:
        data (list): A list of learning scores.
        width (float, optional): The width of the violin plot. Defaults to 0.4.
        group_name (str, optional): The name of the group. Defaults to None.
        proportion (float, optional): The proportion of data used, for labeling. Defaults to None.
        export_path (str, optional): Path to save the plot. Defaults to None.
        verbose (bool, optional): If True, prints summary statistics. Defaults to True.
    """
    # summary stats
    mean_val = np.mean(data)
    se_val   = np.std(data)/np.sqrt(len(data))
    if group_name is None:
        group_name = 'Group'
    if verbose:
        print(f'{group_name} Size: {len(data)}   Avg: {mean_val:.3f}   SE: {se_val:.3f}')

    fig, ax = plt.subplots(figsize=(3,6))
    x = 0.5

    # violin
    parts = ax.violinplot([data],
                          positions=[x],
                          widths=width,
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)
    for violin in parts['bodies']:
        violin.set_facecolor('lightblue')
        violin.set_edgecolor('black')
        violin.set_alpha(0.8)

    # inset boxplot
    ax.boxplot([data],
               positions=[x],
               widths=width*0.5,
               showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))

    # scatter
    jitter = width/8
    x_vals = x + np.random.uniform(-jitter, jitter, size=len(data))
    ax.scatter(x_vals, data, marker='o', zorder=3, color='#1405eb', alpha=0.8)

    # formatting
    ax.set_xlim(0,1)
    ax.set_ylim(20, 65)
    ax.set_xticks([x])
    ax.set_xticklabels([group_name])
    ax.set_xlabel('Group', fontsize=14)
    ax.set_ylabel('Learning Score', fontsize=14)
    ax.set_title(f'Learning Score of {group_name} of {proportion} data', fontsize=16)

    # legend
    patch = mpatches.Patch(color='lightblue', alpha=0.8, label=f'{group_name} (n={len(data)})')
    ax.legend(handles=[patch])

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


def graph_learning_results_single(data: list,
                                  width=0.4,
                                  group_name=None,
                                  proportion=None,
                                  export_path=None,
                                  verbose=True):
    """Graphs the learning result (accuracy) for a single group.

    Args:
        data (list): List of accuracy scores.
        width (float, optional): Width of the violin plot. Defaults to 0.4.
        group_name (str, optional): Name of the group. Defaults to None.
        proportion (float, optional): Proportion of data used for labeling. Defaults to None.
        export_path (str, optional): Path to save the plot. Defaults to None.
        verbose (bool, optional): If True, prints summary statistics. Defaults to True.
    """
    mean_val = np.mean(data)
    se_val   = np.std(data)/np.sqrt(len(data))
    if group_name is None:
        group_name = 'Group'
    if verbose:
        print(f'{group_name} Size: {len(data)}   Avg: {mean_val:.3f}   SE: {se_val:.3f}')

    fig, ax = plt.subplots(figsize=(3,6))
    x = 0.5

    # violin
    parts = ax.violinplot([data],
                          positions=[x],
                          widths=width,
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)
    for violin in parts['bodies']:
        violin.set_facecolor('lightblue')
        violin.set_edgecolor('black')
        violin.set_alpha(0.8)

    # inset boxplot
    ax.boxplot([data],
               positions=[x],
               widths=width*0.5,
               showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))

    # scatter
    jitter = width/8
    x_vals = x + np.random.uniform(-jitter, jitter, size=len(data))
    ax.scatter(x_vals, data, marker='o', zorder=3, color='#1405eb', alpha=0.8)

    # formatting
    ax.set_xlim(0,1)
    ax.set_ylim(55, 85)
    ax.set_xticks([x])
    ax.set_xticklabels([group_name])
    ax.set_xlabel('Group', fontsize=14)
    ax.set_ylabel('Mean Accuracy (%)', fontsize=14)
    ax.set_title(f'Learning Result of {group_name} (last {proportion} data)', fontsize=16)

    # legend
    patch = mpatches.Patch(color='lightblue', alpha=0.8, label=f'{group_name} (n={len(data)})')
    ax.legend(handles=[patch])

    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()
    

def plot_learning_score_trend(
    blocks_groups: list,
    group_labels: list = None,
    proportions: np.ndarray = None,
    block_prop: float = 1.0,
    export_path: str = None
):
    """Plots the trend of learning scores as a function of action proportion.

    Args:
        blocks_groups (list): A list of groups, where each group is a list of block lists (one per subject).
        group_labels (list, optional): Names for each group. Defaults to None.
        proportions (np.ndarray, optional): An array of action proportions to evaluate. Defaults to a linspace.
        block_prop (float, optional): The block proportion to pass to the learning_score function. Defaults to 1.0.
        export_path (str, optional): Path to save the figure. Defaults to None.
    """
    # default labels
    if group_labels is None:
        group_labels = [f"G{idx+1}" for idx in range(len(blocks_groups))]
    # default sweep of action proportions (5%,10%,…,100%)
    if proportions is None:
        proportions = np.linspace(0.05, 1.0, 20)

    # a simple palette for up to 4 groups
    palette = ['#425df5', '#f55442', '#42f58c', '#f5e142']

    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)

    for grp_idx, (blocks_list, label) in enumerate(zip(blocks_groups, group_labels)):
        # build an (n_subjects × n_props) array of scores
        score_matrix = np.array([
            [learning_score(blocks, block_prop=block_prop, action_prop=p)
             for p in proportions]
            for blocks in blocks_list
        ])
        # compute mean and SEM across subjects
        mean_scores = score_matrix.mean(axis=0)
        sem_scores  = score_matrix.std(axis=0, ddof=0) / np.sqrt(score_matrix.shape[0])

        color = palette[grp_idx % len(palette)]
        ax.plot(
            proportions,
            mean_scores,
            label=f"{label} (n={len(blocks_list)})",
            color=color,
            linewidth=2
        )
        ax.fill_between(
            proportions,
            mean_scores - sem_scores,
            mean_scores + sem_scores,
            alpha=0.3,
            color=color
        )

    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("Action Proportion", fontsize=18)
    ax.set_ylabel("Learning Score",    fontsize=18)
    ax.set_title("Learning Score vs Action Proportion", fontsize=22)
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    proportions: np.ndarray = None,
    time_threshold: float = 60,
    pellet_threshold: int = 2,
    export_path: str = None
):
    """Plots the trend of the pellet-in-meal ratio as a function of action proportion.

    Args:
        blocks_groups (list[list[pd.DataFrame]]): A list of groups, each containing lists of blocks for subjects.
        group_labels (list[str], optional): Names for each group. Defaults to None.
        proportions (np.ndarray, optional): Proportions to sweep over. Defaults to a linspace.
        time_threshold (float, optional): Time threshold for meal definition. Defaults to 60.
        pellet_threshold (int, optional): Pellet threshold for meal definition. Defaults to 2.
        export_path (str, optional): Path to save the plot. Defaults to None.
    """
    if group_labels is None:
        group_labels = [f"G{g+1}" for g in range(len(blocks_groups))]
    if proportions is None:
        proportions = np.linspace(0.05, 1.0, 20)

    palette = ['#425df5', '#f55442', '#42f58c', '#f5e142']
    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)

    for gi, (blocks_list, label) in enumerate(zip(blocks_groups, group_labels)):
        # build matrix: rows=subjects, cols=proportions
        mat = np.array([
            [
                # average this sample’s block‐wise ratios at proportion p
                np.nanmean([
                    pellet_ratio_for_block(
                        block_df, p,
                        time_threshold=time_threshold,
                        pellet_threshold=pellet_threshold
                    )
                    for block_df in sample_blocks
                ])
                for p in proportions
            ]
            for sample_blocks in blocks_list
        ])

        mean_ratios = np.nanmean(mat, axis=0)
        sem_ratios  = np.nanstd(mat, axis=0, ddof=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))

        c = palette[gi % len(palette)]
        ax.plot(proportions * 100, mean_ratios, label=f"{label} (n={len(blocks_list)})", color=c, linewidth=2)
        ax.fill_between(proportions * 100,
                        mean_ratios - sem_ratios,
                        mean_ratios + sem_ratios,
                        alpha=0.3, color=c)
    print(f'Female Size: {len(mean_ratios)}   Avg: {mean_ratios[-1]:.3f}   SE: {sem_ratios[-1]:.3f}')
    ax.set_xlabel("Action Proportion (%)", fontsize=18)
    ax.set_ylabel("Pellet‐in‐Meal Ratio", fontsize=18)
    ax.set_title("Pellet‐in‐Meal Ratio vs Action Proportion", fontsize=22)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=14, loc='best')
    plt.tight_layout()

    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()
