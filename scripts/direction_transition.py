import os
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tools as tl
import numpy as np
from meals import find_meals_paper, find_first_good_meal

colors = {'Left': 'red', 'Right': 'blue', 'Pellet': 'green'}


def split_data_to_blocks(data_dropped: pd.DataFrame, day=3) -> list:
    """Split dataframe into blocks of same active poke

    Args:
        data_dropped (pd.DataFrame): behavior data

    Returns:
        list: list of blocks (number of switches of active poke)
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
        meals,_ = find_meals_paper(block, meal_config[0], meal_config[1])
        time = round((meals[0][0] - times[0]).total_seconds() / 60, 2) if len(meals) > 0 else 'no meal'
        
        _, first_meal_time = find_first_good_meal(block, 60, 2, 'cnn')
        if first_meal_time is None:
            meal_1_good = round((times[-1] - times[0]).total_seconds() / 60, 2)
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
            'First_Good_Meal_Time': meal_1_good,
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
        'Success_Count', 'Success_Rate','Active_Poke', 'First_Meal_Time', 'First_Good_Meal_Time',
        'Block_Time', 'Incorrect_Pokes', 'Active', 'Pellet_Rate'])

    return data_stats


def first_meal_stats(data_stats: pd.DataFrame, ignore_inactive=False):
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
        
    # print(time_list, total_list)
    avg_ratio = np.mean(good_meal_list/total_list)
    # print(good_meal_list)
    # print(total_list)
    # print(good_meal_list/total_list)
    avg_time = np.mean(time_list)
    avg_good_time = np.median(good_meal_list)
    return avg_ratio, avg_time, avg_good_time


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


def graph_tranition_stats(data_stats: pd.DataFrame, blocks: list, sheet: str, export_path=None):
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

    info = tl.get_bhv_num(sheet)
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
    

def graph_learning_trend_by_activity(data_stats: pd.DataFrame, blocks: list, path: str, 
                                     block_prop=0.6, action_prop=0.5, export_root=None):
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

    if export_root:
        path = os.path.join(export_root, 'Supplementary 5/', path.replace('.', '')+'.svg')
        plt.savefig(path, bbox_inches='tight')
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
    group = group[group['Event'] != 'Pellet']
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


def learning_result(blocks, action_prop=0.25) -> float:
    results = [accuracy(block[int(len(block)*action_prop):]) for block in blocks]
    return np.mean(results)



def graph_learning_score(ctrl: list,
                         exp: list,
                         width=0.4,
                         group_names=None,
                         proportion=None,
                         export_path=None,
                         verbose=True):
    """
    Graph learning score of two groups using violin + box + scatter.
    Colors: lightblue violin / #1405eb scatter for ctrl; yellow violin / #f28211 scatter for exp.
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
    """
    Graph learning score for a single group using violin + box + scatter.
    Colors: lightblue violin / #1405eb scatter.
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
    """
    Graph accuracy of two groups using violin + box + scatter.
    Colors: lightblue violin / #1405eb scatter for ctrl; yellow violin / #f28211 scatter for exp.
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
    ax.set_ylim(45, 75)
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
    """
    Graph accuracy for a single group using violin + box + scatter.
    Colors: lightblue violin / #1405eb scatter.
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
    ax.set_ylim(45, 75)
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