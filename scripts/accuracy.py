import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import read_excel_by_sheet
from tools import get_bhv_num, get_session_time
import numpy as np
from datetime import datetime, timedelta

def read_and_record(path:str, sheet:str, ending_corr:list, learned_time:list):
    df = read_excel_by_sheet(sheet, path, collect_time=True, 
                             cumulative_accuracy=True, remove_trivial=True, 
                             convert_large=True)
    target_time = timedelta(hours=24)
    df['time_diff'] = (df['Time_passed'] - target_time).abs()
    closest_accuracy = df.loc[df['time_diff'].idxmin(), 'Percent_Correct']
    df.drop(columns=['time_diff'], axis='columns')

    ending_corr.append(closest_accuracy)
    learned_time.append(find_first_learned_time(df))
    return df


def graph_cumulative_acc(mice: list, group=None):
    """
    Graph the line plot for cumulative accuracy of certain group of mice

    Parameters:
    mice: list of data pd.Dataframe data of mice in the group
    group: group number to display on the output plot
    """
    plt.figure(figsize=(15, 6), dpi=90)

    cnt = 1
    for each in mice:
        each['Time_passed_hours'] = each['Time_passed'].dt.total_seconds() / 3600
        sns.lineplot(data=each, x='Time_passed_hours', y='Percent_Correct', label=f'M{cnt}')
        cnt += 1
    plt.grid()
    if isinstance(group, str):
        plt.title(f'Changes in Correction Rate for {group} Group', fontsize=24)
    elif isinstance(group, int):
        plt.title(f'Changes in Correction Rate for Group {group}', fontsize=24)

    plt.xlabel('Session Time (hours)', fontsize=16)
    plt.ylabel('Correct Rate (%)', fontsize=16)
    plt.yticks(range(0, 110, 10))
    plt.legend()
    legend = plt.legend(title='Mice', fontsize=10)
    legend.get_title().set_fontsize(12)
    plt.show()


def cumulative_pellets_meals(data: pd.DataFrame, bhv: int, num: int):
    """
    Graph the cumulative pellet counts for the mice from certain group

    Parameters:
    data: input dataframe of certain mice
    bhv: the group number of the mice
    num: the index of mice
    """
    plt.figure(figsize=(15, 6), dpi=90)

    sns.lineplot(data=data, x='Time_passed', y='Cum_Sum', label='M1')

    plt.grid()
    plt.title(f'Cumulative Sum of Pellet for Control Group {bhv} Mice {num}', fontsize=22)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Cumulative Percentage (%)', fontsize=16)
    plt.legend()
    legend = plt.legend(title='Mice', fontsize=10)
    legend.get_title().set_fontsize(12)
    plt.show()


def calculate_accuracy(group: pd.DataFrame):
    """
    Calculate the percent correct(0-100) in a interval of getting correct poke
    """
    if 'Pellet' in group['Event'].values:
        group = group[group['Event'] != 'Pellet']
        
    total_events = len(group)
    matching_events = group[group['Event'] == group['Active_Poke']]
    matching_count = len(matching_events)
    
    if total_events == 0:
        return 0
    else:
        return (matching_count / total_events) * 100


def find_night_index(hourly_labels:list, rev:bool):
    """Find pairs of indices that is between 7 pm and 7 am

    Args:
        hourly_labels (list): list of times in format of 'hour:minute'

    Returns:
        list: list contains pairs of indices in the night
    """
    intervals = []
    in_interval = False
    interval_start_index = None

    for i, time_str in enumerate(hourly_labels):
        current_time = datetime.strptime(time_str, '%H:%M')

        # Determine if the current time is within the night interval (7pm to 7am)
        if current_time.hour >= 19 or current_time.hour < 7:
            if not in_interval:
                # We are starting a new interval
                in_interval = True
                interval_start_index = i
        else:
            if in_interval:
                # We are ending the current interval
                in_interval = False
                intervals.append([interval_start_index, i - 1])
    
    # If the last time in the list is within the interval, close it
    if in_interval:
        intervals.append([interval_start_index, len(hourly_labels) - 1])
    
    hourly_labels = [datetime.strptime(time_str, '%H:%M').hour for time_str in hourly_labels]
    
    if rev:
        result = []
        start = 0
        for interval in intervals:
            if interval[0] > start:
                result.append([start, interval[0] - 1])
            start = interval[1] + 1
        if start <= len(hourly_labels) - 1:
            result.append([start, len(hourly_labels) - 1])
        return result
    return intervals


def find_first_learned_time(data:pd.DataFrame, window_hours=2, accuracy_threshold=0.8):    
    data['is_match'] = (data['Event'] == data['Active_Poke']).astype(int)
    data['cumulative_total'] = data['is_match'].expanding().count()
    data['cumulative_match'] = data['is_match'].cumsum()
    window_timedelta = timedelta(hours=window_hours)

    for start_idx in range(len(data)):
        start_time = data.loc[start_idx, 'Time_passed']
        end_time = start_time + window_timedelta

        window_data = data[(data['Time_passed'] >= start_time) &
                           (data['Time_passed'] < end_time) &
                           (data['Event'] != 'Pellet')]
        total_events = len(window_data)
        matching_events = window_data['is_match'].sum()
        
        if total_events > 0:
            accuracy = matching_events / total_events
            if accuracy > accuracy_threshold:
                return start_time.total_seconds() / 3600  # Return the precise time that meets the condition
    # not reach high accuracy -> return session time
    return data.loc[len(data)-1, 'Time_passed'].total_seconds() / 3600


def graph_avg_accuracy(ctrl:list, exp:list, width=0.4, exp_group_name=None):
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
    plt.bar(x=[1, 2], height=[ctrl_mean, cask_mean], yerr=[ctrl_err, cask_err], capsize=12,
            tick_label=groups, width=width, color=['lightblue', 'yellow'], alpha=0.8,
            zorder=1, label=[f'Control (n = {len(ctrl)})', f'{exp_name} (n = {len(exp)})'])

    x1 = [1] * len(ctrl)
    x2 = [2] * len(exp)
    plt.scatter(x1, ctrl, marker='o', color='blue', zorder=2) 
    plt.scatter(x2, exp, marker='x', color='orange', zorder=2)

    plt.xlabel('Groups', fontsize=14)
    plt.ylabel('Correct Rate (%)', fontsize=14)
    plt.title(f'Average Correct Rate of Control and {exp_name} Groups in FR1', fontsize=16)

    plt.legend()
    plt.show()


def graph_avg_learned_line(ctrl:list, exp:list, width=0.4, exp_group_name=None):
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
    plt.bar(x=[1, 2], height=[ctrl_mean, cask_mean], yerr=[ctrl_err, cask_err], capsize=12,
            tick_label=groups, width=width, color=['lightblue', 'yellow'], alpha=0.8,
            zorder=1, label=[f'Control (n = {len(ctrl)})', f'{exp_name} (n = {len(exp)})'])

    x1 = [1] * len(ctrl)
    x2 = [2] * len(exp)
    plt.scatter(x1, ctrl, marker='o', color='blue', zorder=2) 
    plt.scatter(x2, exp, marker='x', color='orange', zorder=2)

    plt.xlabel('Groups', fontsize=14)
    plt.ylabel('First Learned Time (hours)', fontsize=14)
    plt.title(f'First Learned Line of Control and {exp_name} Groups', fontsize=16)

    plt.legend()
    plt.show()