import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tools as tl
import numpy as np
from meals import find_meals_paper

colors = {'Left': 'red', 'Right': 'blue', 'Pellet': 'green'}


def split_data_to_blocks(data_dropped: pd.DataFrame) -> list:
    """Split dataframe into blocks of same active poke

    Args:
        data_dropped (pd.DataFrame): behavior data

    Returns:
        list: list of blocks (number of switches of active poke)
    """
    data_dropped = data_dropped.dropna()
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
    """Count transitions in a block

    Args:
        sub_frame (pd.DataFrame): the data of each block

    Returns:
        dict: dictionary that contains 5 info we need
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
    """Count number of occurrence of pellets

    Args:
        sub_frame (pd.DataFrame): data of each block

    Returns:
        int: number of pellets
    """
    pellet_count = 0
    
    for _, row in sub_frame.iterrows():
        event = row['Event']
        
        if 'Pellet' in event:
            pellet_count += 1
    
    return pellet_count


def remove_pellet(block: pd.DataFrame) -> pd.DataFrame:
    return block[block['Event'] != 'Pellet']


def get_transition_info(blocks: list, meal_config:list, reverse:bool) -> pd.DataFrame:
    """Get related statistics about each block
    
    Return a data frame with columns Block_Index, Left_to_Left, Left_to_Right,
    Right_to_Right, Right_to_Left, Success_Count, Success_Rate, Active_Poke, Pellet_Rate
    
    Args:
        blocks (list): list of all blocks

    Returns:
        pd.DataFrame: stats
    """
    new_add = []
    inactives = find_inactive_blocks(blocks, reverse=reverse)

    for i, block in enumerate(blocks):
        no_pellet = remove_pellet(block)
        size = len(no_pellet)
        transitions = count_transitions(no_pellet)
        active_poke = block.iloc[0]['Active_Poke']

        times = block['Time'].tolist()
        meals,_,_ = find_meals_paper(block, meal_config[0], meal_config[1])
        time = round((meals[0][0] - times[0]).total_seconds() / 60, 2) if len(meals) > 0 else 'no meal'
        
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
            'Block_Time': round((times[-1] - times[0]).total_seconds() / 60, 2),
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
        'Success_Count', 'Success_Rate','Active_Poke', 'First_Meal_Time', 'Block_Time', 
        'Incorrect_Pokes', 'Active','Pellet_Rate'])

    return data_stats


def first_meal_stats(data_stats: pd.DataFrame, ignore_inactive=False):
    data_stats = data_stats[:-1]
    total_list = data_stats['Block_Time'].to_numpy(dtype=np.float32)
    time_list = np.array([time if type(time) == float else total_list[idx] 
                          for idx, time in enumerate(data_stats['First_Meal_Time'])])

    if ignore_inactive:
        active_idx = [idx for idx, each in data_stats.iterrows() if each['Active']]
        time_list = time_list[active_idx]
        total_list = total_list[active_idx]
        
    # print(time_list, total_list)
    avg_ratio = np.median(time_list/total_list)
    avg_time = np.median(time_list)
    return avg_ratio, avg_time


def find_inactive_blocks(blocks:list, reverse):
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


def graph_tranition_stats(data_stats: pd.DataFrame, blocks: list, path: str):
    """Graph Statistics of each block in transition
    
    Visualize proportion of each transition, the accuracy and active poke of each block

    Args:
        data_stats (pd.DataFrame): calculated statistics from the data
        blocks (list): list of pd.DataFrame that is splitted from the complete data
        path (str): path of original file path to get display info
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

    labels = data_stats['First_Meal_Time']
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
    
    plt.annotate('First Meal Time (min) \n Block Length', 
            xy=(bars1[0].get_x()+0.4, bars1[0].get_height() + 4.2), 
            xytext=(-90, 20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', lw=2),
            fontsize=16,
            color='blue')
        
    ax.set_xlabel('Blocks', fontsize=16)
    ax.set_ylabel('Percentage(%)', color='black', fontsize=16)
    ax2 = ax.twinx()
    l5 = ax2.plot(data_stats['Block_Index'], data_stats['Incorrect_Pokes'], color='red',
                  label='Incorrect Pokes', alpha=0.5, lw=2)
    ax2.set_ylabel('Incorrect Poke Count', fontsize=16)
    
    info = tl.get_bhv_num(path)
    night_blocks = find_inactive_blocks(blocks, reverse=len(info) == 1)

    for block_index in night_blocks:
        ax.axvspan(block_index - 0.5, block_index + 0.5, facecolor='gray', alpha=0.4)

    left_patch = mpatches.Patch(color='pink', alpha=0.5, label='Left Active')
    right_patch = mpatches.Patch(color='lightblue', alpha=0.5, label='Right Active')
    night_patch = mpatches.Patch(color='gray', alpha=0.5, label='Inactive Period')
    
    lines = [l1[0], l2[0], l3[0], l4[0], l5[0], bars1[0], bars2[0], left_patch, right_patch, night_patch]  # Combine the line objects
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
    fig.set_dpi(100)
    ax.grid(alpha=0.5, linestyle='--')
    plt.show()
    

def graph_learning_trend_by_activity(data_stats: pd.DataFrame, blocks: list, path: str, block_prop=0.6, action_prop=0.5):
    """
    Graph Statistics of first 60% block in transition

    Visualize accuracy of each transition, the accuracy and active poke of each block

    Args:
        data_stats (pd.DataFrame): calculated statistics from the data
        blocks (list): list of pd.DataFrame that is split from the complete data
        path (str): path of original file path to get display info
        block_prop (float): proportion of the blocks used (Default is 0.6)
        action_prop (float): proportion of the data in each block used (Default is 0.5) 
    """
    # Calculate cutoff based on block proportion
    cutoff = int(len(data_stats) * block_prop)
    data_stats = data_stats.iloc[:cutoff].reset_index(drop=True)
    cut_blocks = blocks[:cutoff]
    acc_in_block_by_prop = block_accuracy_by_proportion(cut_blocks, action_prop)

    # Identify inactive and active block indices
    info = tl.get_bhv_num(path)
    night_blocks = find_inactive_blocks(cut_blocks, len(info)==1)
    night_blocks = [each-1 for each in night_blocks]
    active_blocks = [i for i in range(len(data_stats)) if i not in night_blocks]

    # Split data_stats and accuracy into active and inactive periods
    data_stats_active = data_stats.iloc[active_blocks]
    data_stats_inactive = data_stats.iloc[night_blocks]
    acc_active = [acc_in_block_by_prop[i] for i in active_blocks]
    acc_inactive = [acc_in_block_by_prop[i] for i in night_blocks]

    fig, (ax_active, ax_inactive) = plt.subplots(2, 1, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3)  # Adjust space between subplots

    colors_active = ['pink' if poke == 'Left' else 'lightblue' for poke in data_stats_active['Active_Poke']]
    ax_active.bar(data_stats_active['Block_Index'], acc_active, color=colors_active, alpha=0.7)
    ax_active.set_xlabel('Blocks', fontsize=12)
    ax_active.set_ylabel('Success Rate (%)', color='black', fontsize=12)
    ax_active.set_title('Accuracy during Active Periods', fontsize=16)
    ax_active.set_xticks(data_stats_active['Block_Index'])
    ax_active.set_yticks(range(0, 81, 20))
    ax_active.grid(alpha=0.5, linestyle='--')

    colors_inactive = ['pink' if poke == 'Left' else 'lightblue' for poke in data_stats_inactive['Active_Poke']]
    ax_inactive.bar(data_stats_inactive['Block_Index'], acc_inactive, color=colors_inactive, alpha=0.7)
    ax_inactive.set_xlabel('Blocks', fontsize=12)
    ax_inactive.set_ylabel('Success Rate (%)', color='black', fontsize=12)
    ax_inactive.set_title('Accuracy during Inactive Periods', fontsize=16)
    ax_inactive.set_xticks(data_stats_inactive['Block_Index'])
    ax_inactive.set_yticks(range(0, 81, 20))
    ax_inactive.grid(alpha=0.5, linestyle='--')

    left_patch = mpatches.Patch(color='pink', alpha=0.5, label='Left Active')
    right_patch = mpatches.Patch(color='lightblue', alpha=0.5, label='Right Active')

    ax_active.legend(handles=[left_patch, right_patch], loc='upper right', fontsize=9)
    ax_inactive.legend(handles=[left_patch, right_patch], loc='upper right', fontsize=9)

    if len(info) == 2:
        fig.suptitle(f'Accuracy by Switch for Group {info[0]} Mouse {info[1]}', fontsize=18)
    else:
        fig.suptitle(f'Accuracy by Switch for Mouse {info[0]}', fontsize=18)

    plt.show()

    
    
def graph_learning_trend(data_stats: pd.DataFrame, blocks: list, path: str, block_prop=0.6, action_prop=0.5):
    """Graph Statistics of first 60% block in transition
    
    Visualize accuracy of each transition, the accuracy and active poke of each block

    Args:
        data_stats (pd.DataFrame): calculated statistics from the data
        blocks (list): list of pd.DataFrame that is splitted from the complete data
        path (str): path of original file path to get display info
        proportion (float): proportion of the blocks used (Default is 0.6)
        action_prop (float): proportion of the data in each block used (Default is 0.5) 
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

    info = tl.get_bhv_num(path)
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
    
    info = tl.get_bhv_num(path)
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
    """
    Calculate the percent correct(0-100) in a interval of getting correct poke
    """ 
    total_events = len(group)
    matching_events = group[group['Event'] == group['Active_Poke']]
    matching_count = len(matching_events)
    
    if total_events == 0:
        return 0
    else:
        return (matching_count / total_events) * 100
    
    
def block_accuracy_by_proportion(blocks: list, proportion: float):
    acc = []
    for block in blocks:
        size = int(len(block) * proportion)
        acc.append(accuracy(block[:size]))
    return acc


def learning_score(blocks: list, block_prop=0.5, action_prop=0.8) -> float:
    """Calculate learning score for each sample
    
    We take first proportion amount of the data to calculate accuracy, 
    which indicates the 

    Args:
        blocks (list): list of all splited blocks of each mouse
        proportion (float): the proportion of the data we use to measure learning (Default=0.5) 

    Returns:
        float: score calculated
    """
    cutoff = int(len(blocks)*block_prop)
    return np.mean(block_accuracy_by_proportion(blocks=blocks[:cutoff], proportion=action_prop))


def graph_learning_score(ctrl:list, exp:list, width=0.4, exp_group_name=None, proportion=None):
    """
    Graph learning score of two groups

    Args:
        ctrl (list): data of control group
        exp (list): data of experiment group
        width (float): width of plotted bars
        exp_group_name (str, Optional): name of the experiment group, name with treatments usually.
        proportion (float): proportion of the data we use to evaluate learning performance
    """
    ctrl_mean = np.mean(ctrl)
    cask_mean = np.mean(exp)
    ctrl_err = np.std(ctrl) / np.sqrt(len(ctrl))
    cask_err = np.std(exp) / np.sqrt(len(exp))

    exp_name = 'Experiment' if exp_group_name==None else exp_group_name
    groups = ['Control', exp_name]

    plt.figure(figsize=(7, 7))
    plt.bar([1, 2], [ctrl_mean, cask_mean], yerr=[ctrl_err, cask_err], capsize=12, tick_label=groups, 
            width=width, color=['lightblue', 'yellow'], alpha=0.8, zorder=1, label=['Control', exp_name])

    x1 = [1] * len(ctrl)
    x2 = [2] * len(exp)
    plt.scatter(x1, ctrl, marker='o', color='blue', zorder=2) 
    plt.scatter(x2, exp, marker='x', color='orange', zorder=2)

    plt.xlabel('Groups', fontsize=14)
    plt.ylabel('Learning Score', fontsize=14)
    plt.title(f'Learning Score Control and {exp_name} Groups with {proportion} Data', fontsize=16)

    plt.legend()
    plt.show()
        

def graph_retrieval_time(ctrl:list, exp:list, width=0.4, exp_group_name=None, rev=False):
    """
    Graph average correct rate

    Args:
        ctrl (list): data of control group
        exp (list): data of experiment group
        width (float): width of plotted bars
        exp_group_name (str, Optional): name of the experiment group, name with treatments usually.
    """
    ctrl_mean = np.mean(ctrl)
    cask_mean = np.mean(exp)
    ctrl_err = np.std(ctrl) / np.sqrt(len(ctrl))
    cask_err = np.std(exp) / np.sqrt(len(exp))

    exp_name = 'Experiment' if exp_group_name==None else exp_group_name
    groups = ['Control', exp_name]

    plt.figure(figsize=(7, 7))
    plt.bar([1, 2], [ctrl_mean, cask_mean], yerr=[ctrl_err, cask_err], capsize=12, tick_label=groups, 
            width=width, color=['lightblue', 'yellow'], alpha=0.8, zorder=1, label=['Control', exp_name])

    x1 = [1] * len(ctrl)
    x2 = [2] * len(exp)
    plt.scatter(x1, ctrl, marker='o', color='blue', zorder=2) 
    plt.scatter(x2, exp, marker='x', color='orange', zorder=2)

    plt.xlabel('Groups', fontsize=14)
    plt.ylabel('Time (minutes)', fontsize=14)
    exp_type = 'Reversal' if rev else 'FR1'
    plt.title(f'Pellet Retrieval Time of Control and {exp_name} Groups in {exp_type}', fontsize=16)
    plt.legend()
    plt.show()
    

def graph_slope(ctrl:list, exp:list, width=0.4, exp_group_name=None, rev=False):
    """
    Graph average correct rate

    Args:
        ctrl (list): data of control group
        exp (list): data of experiment group
        width (float): width of plotted bars
        exp_group_name (str, Optional): name of the experiment group, name with treatments usually.
    """
    ctrl_mean = np.mean(ctrl)
    cask_mean = np.mean(exp)
    ctrl_err = np.std(ctrl) / np.sqrt(len(ctrl))
    cask_err = np.std(exp) / np.sqrt(len(exp))

    exp_name = 'Experiment' if exp_group_name==None else exp_group_name
    groups = ['Control', exp_name]

    plt.figure(figsize=(7, 7))
    plt.bar([1, 2], [ctrl_mean, cask_mean], yerr=[ctrl_err, cask_err], capsize=12, tick_label=groups, 
            width=width, color=['lightblue', 'yellow'], alpha=0.8, zorder=1, label=['Control', exp_name])

    x1 = [1] * len(ctrl)
    x2 = [2] * len(exp)
    plt.scatter(x1, ctrl, marker='o', color='blue', zorder=2) 
    plt.scatter(x2, exp, marker='x', color='orange', zorder=2)

    plt.xlabel('Groups', fontsize=14)
    plt.ylabel('Time (minutes)', fontsize=14)
    exp_type = 'Reversal' if rev else 'FR1'
    plt.title(f'Slope of Retrieval Time Best-Fit Line of Control and {exp_name} Groups in {exp_type}', fontsize=16)
    plt.legend()
    plt.show()