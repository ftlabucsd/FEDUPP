import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tools import get_bhv_num
from preprocessing import get_retrieval_time, read_excel_by_sheet
import numpy as np
from direction_transition import split_data_to_blocks
import os


def count_interval(data: pd.DataFrame) -> list:
    """Get intervals in minutes between each two actions in a list

    Args:
        data (pd.DataFrame): behavior data

    Returns:
        list: list of intervals
    """
    intervals = []
    
    for i in range(1, len(data)):
        current_timestamp = data.iloc[i]['Time']
        previous_timestamp = data.iloc[i - 1]['Time']
        
        interval = (current_timestamp - previous_timestamp).total_seconds() / 60
        intervals.append(interval)
    
    return intervals


def clean_and_interval(path: str) -> pd.DataFrame:
    """Add interval column to the data

    Interval column means the time interval between current action and previous action
    in terms of minutes
    
    Args:
        data (pd.DataFrame): raw readed data

    Returns:
        pd.DataFrame: data with interval column
    """
    data = pd.read_csv(path)
    data = data[['MM:DD:YYYY hh:mm:ss', 'Event', 'Retrieval_Time']].rename(columns={'MM:DD:YYYY hh:mm:ss' : 'Time', 
                                                                                    'Retrieval_Time': 'collect_time'})
    data = data[data['Event'] == 'Pellet'].reset_index().drop('index', axis='columns')
    data['Time'] = pd.to_datetime(data['Time'])

    # calculate time
    data['Interval'] = data['Time'].diff().fillna(pd.Timedelta(seconds=0))
    data['Interval'] = data['Interval'].dt.total_seconds() / 60

    return data


def graph_pellet_interval(path: str):
    """Graph Intervals of actions with respect to time

    Args:
        path (str): filepath of csv
    """
    data = clean_and_interval(path)
    
    plt.figure(figsize=(15, 5))

    sns.set_palette('bright')
    sns.set_style('darkgrid')

    sns.lineplot(data=data, x='Time', y='Interval', alpha=0.8)

    info = get_bhv_num(path)
    plt.title(f'Interval Between Pellets for Group {info[0]} Mouse {info[1]}', fontsize=18)
    plt.xlabel('Time')
    plt.ylabel('Interval (minutes)')
    plt.show()
    
    
def mean_pellet_collect_time(path:str, sheet:str, remove_outlier=False, n_stds=3, day=3):
    pellet_times = get_retrieval_time(path, sheet, day=day)
    mean = np.mean(pellet_times)
    std = np.std(pellet_times)
    if remove_outlier:
        cutoff = mean+std*n_stds
        pellet_times = [each for each in pellet_times if each < cutoff]
    return pellet_times, np.mean(pellet_times), np.std(pellet_times)


def plot_retrieval_time_by_block(path:str, sheet:str, day=3, n_stds=3, export_path=None):
    pellet_times = get_retrieval_time(path, sheet, day=10)
    mean = np.mean(pellet_times)
    std = np.std(pellet_times)
    cutoff = mean+std*n_stds

    time_by_block = []
    data = read_excel_by_sheet(sheet, path, collect_time=True)
    blocks = split_data_to_blocks(data, day=day)
    for block in blocks:
        times = block['collect_time'].tolist()
        times = [each for each in times if each != 0 and each < cutoff]
        time_by_block.append(np.mean(times) if len(times) != 0 else 0)

    temp = time_by_block[:-1]
    block_indices = np.arange(len(temp))
    slope, intercept = np.polyfit(block_indices, temp, 1)
    best_fit_line = slope * block_indices + intercept

    plt.figure(figsize=(6, 4), dpi=120)
    plt.plot(time_by_block, marker='*')
    plt.plot(block_indices, best_fit_line, color='red', linestyle='--', 
             alpha=0.75, label=f'Best Fit Line (slope: {slope:.2f})')
    plt.xlabel('Blocks', fontsize=14)
    plt.ylabel('Mean Time (min)', fontsize=14)
    
    info = get_bhv_num(sheet)
    plt.title(f'Retrieval Time of Group {info[0]} Mouse {info[1]}', fontsize=18)
    plt.grid()
    plt.legend()

    if export_path:
        plt.savefig(export_path, bbox_inches='tight')

    plt.show()
    return time_by_block, best_fit_line[-1]+slope, round(slope, 2)
    
    
def perform_T_test(ctrl:list, exp:list, test_side='two-sided', alpha=0.05, paired=False):
    """Perform T tests on control and experiment groups

    Args:
        ctrl (list): data from control group
        exp (list): data from experiment group
        test_side (str): the alternative hypothesis 
                two-sided: not equal
                greater: exp mean > ctrl mean
                less: exp mean < ctrl mean
        alpha (float, optional): significance level of the test. Defaults to 0.05.
        paired (bool): if true, it means two groups are paired data, while false means 
            two independent sets. Defaults to 
    """
    if test_side not in ['two-sided', 'less', 'greater']:
        print('Test size must be two-sided, less or greater')
        return
    
    if paired:
        _, p_value = stats.ttest_rel(exp, ctrl, alternative=test_side)
    else:
        _, p_value = stats.ttest_ind(exp, ctrl, alternative=test_side)

    print("P Value is ", p_value)
    if p_value < alpha:
        if test_side == 'two-sided':
            print("There is a significant difference between the two groups.")
        else:
            print(f'Experiment group is significantly {test_side} than control group')
    else:
        print("There is no significant difference between the two groups.")


def graph_retrieval_time(ctrl:list, exp:list, width=0.4, exp_group_name=None):
    """
    Graph average retrieval time of pellets

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
    plt.bar(x=[1, 2], height=[ctrl_mean, cask_mean], yerr=[ctrl_err, cask_err], capsize=12,
            tick_label=groups, width=width, color=['lightblue', 'yellow'], alpha=0.8,
            zorder=1, label=[f'Control (n = {len(ctrl)})', f'{exp_name} (n = {len(exp)})'])
    
    x1 = [1] * len(ctrl)
    x2 = [2] * len(exp)
    plt.scatter(x1, ctrl, marker='o', color='blue', zorder=2) 
    plt.scatter(x2, exp, marker='x', color='orange', zorder=2)

    plt.xlabel('Groups', fontsize=14)
    plt.ylabel('Retrieval Time (min)', fontsize=14)
    plt.title(f'Average  of Control and {exp_name} Groups in FR1', fontsize=16)

    plt.legend()
    plt.show()
