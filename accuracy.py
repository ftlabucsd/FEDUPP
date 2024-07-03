import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import read_excel_by_sheet, read_csv_clean
from tools import get_bhv_num, get_session_time
import numpy as np
from datetime import datetime

def graph_cumulative_acc(mice: list, group: int):
    """
    Graph the line plot for cumulative accuracy of certain group of mice

    Parameters:
    mice: list of data pd.Dataframe data of mice in the group
    group: group number to display on the output plot
    """
    plt.figure(figsize=(15, 6), dpi=90)

    cnt = 1
    for each in mice:
        sns.lineplot(data=each, x='Time', y='Percent_Correct', label=f'M{cnt}')
        cnt += 1
    plt.grid()
    plt.title(f'Changes in Correction Rate for Control Group {group}', fontsize=24)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Correct Rate', fontsize=16)
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

    sns.lineplot(data=data, x='Time', y='Cum_Sum', label='M1')

    plt.grid()
    plt.title(f'Cumulative Sum of Pellet for Control Group {bhv} Mice {num}', fontsize=22)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Cumulative Percentage', fontsize=16)
    plt.legend()
    legend = plt.legend(title='Mice', fontsize=10)
    legend.get_title().set_fontsize(12)
    plt.show()


def calculate_accuracy(group):
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


def instant_acc(sheet=None, parent='../behavior data integrated/Adjusted FED3 Data.xlsx', path=None, csv=False):
    if csv:
        df = read_csv_clean(path=path)
    else:
        df = read_excel_by_sheet(sheet, parent, hundredize=False,
                                convert_time=True, remove_trival=False)
    
    df = df[df['Event'] != 'Pellet'].reset_index()

    # remove possible enomaly value
    if (df['Time'].loc[1] - df['Time'].loc[0]).total_seconds() / 3600 > 2:
        df = df[1:].reset_index()

    # check pellet over-consumption
    if not csv: # only check when dealing with excel
        idx = 0
        for each in df.itertuples():
            idx += 1
            if each[-1] > 1:
                break
        df = df[:idx]
    
    # Resample the data to hourly intervals and apply the accuracy calculation function
    duration = get_session_time(df)
    df.set_index('Time', inplace=True)
    
    # choose interval based on total duration
    if duration < 50:
        result = df.resample('1h').apply(calculate_accuracy).reset_index().rename(columns={0: 'Accuracy'})
    elif 50 <= duration < 100:
        result = df.resample('2h').apply(calculate_accuracy).reset_index().rename(columns={0: 'Accuracy'})
    elif 100 <= duration < 300:
        result = df.resample('4h').apply(calculate_accuracy).reset_index().rename(columns={0: 'Accuracy'})
    else:
        result = df.resample('10h').apply(calculate_accuracy).reset_index().rename(columns={0: 'Accuracy'})
        
    result['Time'] = pd.to_datetime(result['Time'])
    
    if csv:
        return result

    return result, get_bhv_num(sheet)


def find_night_index(hourly_labels:list):
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
    
    return intervals


def graph_instant_acc(data, bhv, num, lr_time):
    plt.figure(figsize=(13, 7), dpi=90)

    ax = sns.barplot(x=data['Time'], y=data['Accuracy'], color="skyblue", width=0.6, label='Accuracy')

    xtick_positions = ax.get_xticks()

    hourly_labels = [label.strftime('%H:%M') for label in data['Time'] if label.minute == 0]
    hourly_positions = [pos for pos, label in zip(xtick_positions, data['Time']) if label.minute == 0]
    ax.set_xticks(hourly_positions)  # Set the tick positions to match the hourly intervals
    ax.set_xticklabels(hourly_labels, rotation=45, horizontalalignment='right')  # Set the tick labels to hourly format
    
    # Locate the x-coordinates for the specified times
    dark = find_night_index(hourly_labels)
    
    for idx, each in enumerate(dark):
        if idx == 0:
            ax.axvspan(each[0], each[1], color='grey', alpha=0.4, label='Night')
        else:
            ax.axvspan(each[0], each[1], color='grey', alpha=0.4)

    if lr_time != None:
        lr_time = lr_time.strftime('%H:%M')
        i = hourly_positions[hourly_labels.index(lr_time)]
        plt.axvline(i, color='red', label='1st Learned')
    plt.xlabel('Time')
    plt.ylabel('Accuracy (%)')
    if bhv == None:
        plt.title('Accuracy over Time')
    else:
        plt.title(f'Accuracy over Time of Group {bhv} Mice {num}')
    plt.legend()
    plt.show()


def time_high_acc(grouped_data: pd.DataFrame):
    """
    return 1st time we have 2 continuous hours with >=80% accuracy
    """ 
    first_time = None
    res = False
    time_taken = -1
    for idx, row in grouped_data.iterrows():
        if idx == 0: continue
        if row['Accuracy'] >= 80 and grouped_data.loc[idx-1]['Accuracy'] >= 80:
            res = True
            first_time = grouped_data.loc[idx-1]['Time']
            time_taken = (first_time - grouped_data['Time'].min()).total_seconds() / 3600
            break
    return first_time, time_taken if res else None


def graph_avg_corr_rate(ctrl:list, exp:list, width=0.4, exp_group_name=None):
    ctrl_mean = np.mean(ctrl)
    cask_mean = np.mean(exp)
    ctrl_err = np.std(ctrl) / np.sqrt(len(ctrl))
    cask_err = np.std(exp) / np.sqrt(len(exp))
    
    exp_name = 'Experiment' if exp_group_name==None else exp_group_name
    groups = ['Control', exp_name]
    
    plt.figure(figsize=(7, 7))
    plt.bar([1, 2], [ctrl_mean, cask_mean], yerr=[ctrl_err, cask_err], capsize=12, tick_label=groups, 
            width=width, color=['lightblue', 'yellow'], alpha=0.8, zorder=1, label=['Control', 'CASK'])

    x1 = [1] * len(ctrl)
    x2 = [2] * len(exp)
    plt.scatter(x1, ctrl, marker='o', color='blue', zorder=2) 
    plt.scatter(x2, exp, marker='x', color='orange', zorder=2)

    plt.xlabel('Groups', fontsize=14)
    plt.ylabel('Correct Rate (%)', fontsize=14)
    plt.title(f'Average Correct Rate of Control and {exp_name} Groups in FR1', fontsize=16)

    plt.legend()
    plt.show()
