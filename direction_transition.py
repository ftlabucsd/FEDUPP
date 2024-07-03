import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tools as tl
import seaborn as sns
from preprocessing import csv_read_clean, excel_read_clean

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
            blocks.append(data_dropped.iloc[start_idx:key])  # add current block
            start_idx = key # update start_idx
            curr_poke = val['Active_Poke']   # update poke marker

    blocks.append(data_dropped.iloc[start_idx:])  # append the last block
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


def get_transition_info(blocks: list) -> pd.DataFrame:
    """Get related statistics about each block
    
    Return a data frame with columns Block_Index, Left_to_Left, Left_to_Right,
    Right_to_Right, Right_to_Left, Success_Count, Success_Rate, Active_Poke, Pellet_Rate
    
    Args:
        blocks (list): list of all blocks

    Returns:
        pd.DataFrame: stats
    """
    new_add = []

    for i, block in enumerate(blocks):
        no_pellet = remove_pellet(block)
        size = len(no_pellet)
        transitions = count_transitions(no_pellet)
        active_poke = block.iloc[0]['Active_Poke']
        new_row_data = {
            'Block_Index': i+1,
            'Left_to_Left': round(transitions.get('Left_to_Left')/size * 100, 2),
            'Left_to_Right': round(transitions.get('Left_to_Right')/size * 100, 2),
            'Right_to_Right': round(transitions.get('Right_to_Right')/size * 100, 2),
            'Right_to_Left': round(transitions.get('Right_to_Left')/size * 100, 2),
            'Success_Count': transitions.get('success_count'),
            'Success_Rate' : round(transitions.get('success_count')/size * 100, 2),
            'Active_Poke' : active_poke,
            'Total_Count': size
        }
        new_add.append(new_row_data)

    idx = 0
    for each in new_add:
        count = count_pellet(blocks[idx])
        each['Pellet_Rate'] = round(count / len(blocks[idx]), 2)
        idx += 1
        

    data_stats = pd.DataFrame(new_add, columns=[
        'Block_Index', 'Left_to_Left', 'Left_to_Right', 'Right_to_Right', 'Right_to_Left',
        'Success_Count', 'Success_Rate','Active_Poke', 'Pellet_Rate'])

    return data_stats


def graph_tranition_stats(data_stats: pd.DataFrame, blocks: list, path: str):
    fig, ax = plt.subplots(figsize=(22, 12))
    bhv, num = tl.get_bhv_num(path)


    ax.plot(data_stats['Block_Index'], data_stats['Left_to_Left'],
            marker='o', label='Left-Left', color='black')
    ax.plot(data_stats['Block_Index'], data_stats['Left_to_Right'],
            marker='*', label='Left-Right', color='green')
    ax.plot(data_stats['Block_Index'], data_stats['Right_to_Right'],
            marker='s', label='Right-Right', color='orange')
    ax.plot(data_stats['Block_Index'], data_stats['Right_to_Left'],
            marker='X', label='Right-Left', color='red')

    legend = plt.legend(title='Poke Transitions', loc='upper right')

    ax.bar(data_stats['Block_Index'][::2], data_stats['Success_Rate'][::2],
        label='Success Rate', color='pink' if data_stats['Active_Poke'][0] == 'Left' else 'lightblue', alpha=0.7)
    ax.bar(data_stats['Block_Index'][1::2], data_stats['Success_Rate'][1::2],
        label='Success Rate', color='lightblue' if data_stats['Active_Poke'][1] == 'Right' else 'pink', alpha=0.7)

    ax.set_xlabel('Blocks', fontsize=16)
    ax.set_ylabel('Percentage(%)', color='black', fontsize=16)

    legend.get_title().set_fontsize('17')
    for text in legend.get_texts():
        text.set_fontsize('15')

    night_blocks = []
    block_start_index = 1

    for block_df in blocks:
        if not block_df.empty and 'Time' in block_df:
            first_timestamp = pd.to_datetime(block_df['Time'].iloc[0])
            if 19 <= first_timestamp.hour or first_timestamp.hour < 7:
                night_blocks.append(block_start_index)
        block_start_index += 1

    for block_index in night_blocks:
        ax.axvspan(block_index - 0.5, block_index + 0.5, facecolor='gray', alpha=0.4)

    left_patch = mpatches.Patch(color='pink', alpha=0.5, label='Left Active')
    right_patch = mpatches.Patch(color='lightblue', alpha=0.5, label='Right Active')
    night_patch = mpatches.Patch(color='gray', alpha=0.5, label='Night Period')

    leg_bg = plt.legend(handles=[left_patch, right_patch, night_patch], loc='upper right', bbox_to_anchor=(1.0, 0.84))

    leg_bg.set_title('Correct Rate')
    leg_bg.get_title().set_fontsize('17')
    leg_bg.get_texts()[0].set_fontsize('15')
    leg_bg.get_texts()[1].set_fontsize('15')
    leg_bg.get_texts()[2].set_fontsize('15')

    ax = plt.gca()
    ax.add_artist(legend)

    plt.title(f'Probability of Transitions in Poke Choosing and Correct Rates in Group {bhv} Mice {num}', fontsize=24)
    plt.xticks(data_stats['Block_Index'])
    plt.yticks(range(0, 100, 20))
    fig.set_dpi(80)
    plt.grid(alpha=0.5, linestyle='--')
    plt.show()

def graph_diff(diff: pd.DataFrame):
    plt.figure(figsize=(22, 12))
    sns.lineplot(data=diff, x='Block_Index', y='Difference')
    plt.xticks(range(1, len(diff['Block_Index']) + 1))
    plt.title('Difference of the Left and Right Sticking', fontsize=18)
    plt.grid()
    plt.show()

def get_difference_key(data_stats: pd.DataFrame) -> (pd.DataFrame, bool):
    diff = pd.DataFrame(data=data_stats[['Block_Index', 'Left_to_Left', 'Right_to_Right']])
    diff['Left_to_Left'] -= diff['Right_to_Right']
    diff = diff.drop(['Right_to_Right'], axis='columns').rename(columns={'Left_to_Left':'Difference'})
    return diff, data_stats['Active_Poke'][0] == 'Left'
