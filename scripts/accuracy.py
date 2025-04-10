import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import read_excel_by_sheet, preprocess_csv
from tools import get_bhv_num, get_session_time
import numpy as np
from datetime import datetime, timedelta
import matplotlib.patches as mpatches

def read_and_record(path:str, sheet:str, ending_corr:list, learned_time:list, acc_dict:dict):
    df = read_excel_by_sheet(sheet, path, collect_time=True, 
                             cumulative_accuracy=True, remove_trivial=True, 
                             convert_large=True)
    target_time = timedelta(hours=24)
    df['time_diff'] = (df['Time_passed'] - target_time).abs()
    closest_accuracy = df.loc[df['time_diff'].idxmin(), 'Percent_Correct']
    df = df[:df['time_diff'].idxmin()]
    df.drop(columns=['time_diff'], axis='columns')

    ending_corr.append(closest_accuracy)
    learned_time.append(find_first_learned_time(df))
    acc_dict[sheet] = closest_accuracy
    return df

def rr_csv(csv_path, ending_corr:list, learned_time:list):
    df = preprocess_csv(csv_path, collect_time=True, 
                             cumulative_accuracy=True, remove_trivial=True, 
                             convert_large=True)
    target_time = timedelta(days=7)
    df['time_diff'] = (df['Time_passed'] - target_time).abs()
    closest_accuracy = df.loc[df['time_diff'].idxmin(), 'Percent_Correct']
    df = df[:df['time_diff'].idxmin()]
    df.drop(columns=['time_diff'], axis='columns')

    ending_corr.append(closest_accuracy)
    learned_time.append(find_first_learned_time(df))
    return df


def graph_cumulative_acc(mice: list, group=None, export_path=None):
    """
    Graph the line plot for cumulative accuracy of certain group of mice

    Parameters:
    mice: list of data pd.Dataframe data of mice in the group
    group: group number to display on the output plot
    """
    plt.figure(figsize=(15, 5), dpi=150)

    cnt = 1
    for each in mice:
        each['Time_passed_hours'] = each['Time_passed'].dt.total_seconds() / 3600
        sns.lineplot(data=each, x='Time_passed_hours', y='Percent_Correct', label=f'M{cnt}')
        cnt += 1
    plt.grid()
    plt.title(f'Cumulative Accuracy for Cohort {group}', fontsize=24)
    plt.xlabel('Session Time (hours)', fontsize=16)
    plt.ylabel('Correct Rate (%)', fontsize=16)
    plt.yticks(range(0, 110, 10))
    plt.legend()
    legend = plt.legend(title='Mice', fontsize=10)
    legend.get_title().set_fontsize(12)
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
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
    data = data[data['Event'] != 'Pellet'].reset_index(drop=True)
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


def graph_group_stats(ctrl: list, exp: list, stats_name: str, unit: str, 
                      violin_width=0.25, dpi=150, group_names=None, verbose=True, export_path=None):
    # Set default group names if not provided
    if group_names is None or len(group_names) < 2:
        group_names = ['Control', 'Experiment']
    ctrl_name, exp_name = group_names

    # Calculate summary statistics (for logging purposes)
    if verbose:
        ctrl_averages = np.mean(ctrl)
        exp_averages = np.mean(exp)
        ctrl_std = np.std(ctrl) / np.sqrt(len(ctrl))
        exp_std = np.std(exp) / np.sqrt(len(exp))
        print(f'{ctrl_name} Size: {len(ctrl)}')
        print(f'{exp_name} Size: {len(exp)}')
        print(f'{ctrl_name} Average: {ctrl_averages}')
        print(f'{exp_name} Average: {exp_averages}')
        print(f'{ctrl_name} Standard Error: {ctrl_std}')
        print(f'{exp_name} Standard Error: {exp_std}')

    # Create the figure and axis
    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(6, 6)
    
    # Define x positions for the groups. (You can adjust these values as needed.)
    x_positions = [0.5, 1.0]
    
    # Create the violin plots. Note that violinplot expects a list of datasets.
    data = [ctrl, exp]
    parts = ax.violinplot(data, positions=x_positions, widths=violin_width, 
                            showmeans=False, showmedians=False, showextrema=False)
    ax.scatter([x_positions[0]]*len(ctrl), ctrl, color='#18c1f5')
    ax.scatter([x_positions[1]]*len(exp), exp, color='#36d15a')
    
    # Customize the appearance of each violin using the returned 'bodies'
    for i, violin in enumerate(parts['bodies']):
        color = '#38bcf5' if i == 0 else '#38f5a6'
        violin.set_facecolor(color)
        violin.set_edgecolor('black')
        violin.set_alpha(0.6)

    # Create custom legend patches.
    ctrl_patch = mpatches.Patch(color='#38bcf5', alpha=0.6, label=f'{ctrl_name} (n = {len(ctrl)})')
    exp_patch = mpatches.Patch(color='#38f5a6', alpha=0.6, label=f'{exp_name} (n = {len(exp)})')
    ax.legend(handles=[ctrl_patch, exp_patch])
    
    # Set the labels and title.
    ax.set_xlabel('Groups', fontsize=14)
    # The y-axis label now represents the measured statistic's units.
    ax.set_ylabel(f'{stats_name} ({unit})', fontsize=14)
    ax.set_title(f'{stats_name} Distribution for {ctrl_name} and {exp_name} Groups', fontsize=20)
    
    # Set the x-ticks and tick labels.
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names)
    
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()


def graph_single_stats(data: list, stats_name: str, unit: str, 
                       violin_width=0.25, dpi=150, group_name=None, verbose=True, export_path=None):
    # Calculate summary statistics (for logging purposes)
    average = np.mean(data)
    std_error = np.std(data) / np.sqrt(len(data))

    if group_name is None:
        group_name = 'Group'
    
    if verbose:
        print(f'{group_name} Size: {len(data)}')
        print(f'{group_name} Average: {average}')
        print(f'{group_name} Standard Error: {std_error}')
    
    # Create the figure and axis
    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(6, 6)
    
    # Define the x position for the group (only one group, so a single x value)
    x_positions = [0.5]
    
    # Create the violin plot for the single dataset.
    # Note that violinplot expects a list of datasets.
    parts = ax.violinplot([data], positions=x_positions, widths=violin_width, 
                            showmeans=False, showmedians=False, showextrema=False)
    
    # Overlay individual data points as a scatter plot
    ax.scatter([x_positions[0]] * len(data), data, marker='o', zorder=2, color='#18c1f5', alpha=0.8)

    # Customize the appearance of the violin plot. Since we only have one violin, we assign one color.
    for i, violin in enumerate(parts['bodies']):
        color = '#38bcf5'
        violin.set_facecolor(color)
        violin.set_edgecolor('black')
        violin.set_alpha(0.6)
    
    # Create a custom legend patch for the single group.
    group_patch = mpatches.Patch(color='#38bcf5', alpha=0.6, label=f'{group_name} (n = {len(data)})')
    ax.legend(handles=[group_patch])
    
    # Labeling the axes and the plot title.
    ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylabel(f'{stats_name} ({unit})', fontsize=14)
    ax.set_title(f'{stats_name} Distribution for {group_name}', fontsize=20)
    
    # Set the x-ticks and labels.
    ax.set_xticks(x_positions)
    ax.set_xticklabels([group_name])
    
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()
