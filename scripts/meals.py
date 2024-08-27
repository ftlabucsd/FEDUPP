import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from datetime import timedelta
import numpy as np
from accuracy import find_night_index

plt.rcParams['figure.figsize'] = (20, 6)


def pellet_flip(data: pd.DataFrame) -> pd.DataFrame:
    """return a dataframe with 10-min interval and pellet count in this interval

    Args:
        data (pd.DataFrame): raw dataframe processed with pipeline

    Returns:
        pd.DataFrame: data with 10-min interval and pellet count
    """
    data = data.set_index('Time')
    grouped_data = data[data['Event'] == 'Pellet'].resample('10min').size().reset_index()
    grouped_data.columns = ['Interval_Start', 'Pellet_Count']
    
    return grouped_data


def average_pellet(group: pd.DataFrame) -> float:
    """return average hour pellet count in a session

    Args:
        group (pd.DataFrame): data processed by pellet flip

    Returns:
        float: average pellet count
    """
    total_hr = (group['Interval_Start'].max()-group['Interval_Start'].min()).total_seconds() / 3600
    total_pellet = group['Pellet_Count'].sum()
    return round(total_pellet / total_hr, 3)


def find_pellet_frequency(data: pd.DataFrame) -> pd.DataFrame:
    """find number of pellet in every 10 minutes
    return a new data frame records the 10 minutes pellet
    """
    data = data.drop(['Pellet_Count'], axis='columns')
    data.set_index('Time', inplace=True)
    grouped_data = data[data['Event'] == 'Pellet'].resample('10min').size().reset_index()
    grouped_data.columns = ['Interval_Start', 'Pellet_Count']
    
    return grouped_data


def graph_pellet_frequency(grouped_data: pd.DataFrame, bhv, num):
    """graph histogram for pellet frequency
    histogram analysis
    """
    ax = sns.barplot(data=grouped_data, x='Interval_Start', y='Pellet_Count', color='#000099', alpha=0.5)

    # Get the x-axis positions
    xtick_positions = ax.get_xticks()

    # Update x-axis labels to display one-hour intervals
    hourly_labels = [label.strftime('%H:%M') for label in grouped_data['Interval_Start'] if label.minute == 0]
    hourly_positions = [pos for pos, label in zip(xtick_positions, grouped_data['Interval_Start']) if label.minute == 0]

    ax.set_xticks(hourly_positions)  # Set the tick positions to match the hourly intervals
    ax.set_xticklabels(hourly_labels, rotation=45, horizontalalignment='right')  # Set the tick labels to hourly format
    
    # Locate the x-coordinates for the specified times
    if bhv is not None:
        dark = find_night_index(hourly_labels, rev=False)
    else:
        dark = find_night_index(hourly_labels, rev=True)

    for idx, each in enumerate(dark):
        if idx == 0:
            ax.axvspan(6*each[0], 6*(1+each[1]), color='grey', alpha=0.4, label='Inactive')
        else:
            ax.axvspan(6*each[0], 6*(1+each[1]), color='grey', alpha=0.4)

    # Add vertical grey background for the time interval between 7 p.m. and 7 a.m.
    plt.axhline(y=5, color='red', linestyle='--', label='meal')
    if bhv == None:
        plt.title(f'Pellet Frequency of Mouse {num}', fontsize=18)
    else:
        plt.title(f'Pellet Frequency of Group {bhv} Mouse {num}', fontsize=18)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Number of Pellet', fontsize=14)
    plt.yticks(range(0, 19, 2))
    plt.tight_layout()
    plt.legend()
    plt.show()


def meal_threshold(data: pd.DataFrame, collect_quantile=0.6, pellet_quantile=0.75) -> tuple:
    data = data[data['Event'] == 'Pellet'].reset_index().drop('index', axis='columns')
    data['Time'] = pd.to_datetime(data['Time'])

    data['Interval'] = data['Time'].diff().fillna(pd.Timedelta(seconds=0))
    data['Interval'] = data['Interval'].dt.total_seconds() / 60
    
    data['collect_time'] = pd.to_numeric(data['collect_time'], errors='coerce')
    max_value = data['collect_time'].max()
    data['collect_time'] = data['collect_time'].replace('Timed_out', max_value)

    collect_time_thres = data['collect_time'].quantile(collect_quantile)
    pellet_interval_thres = data['Interval'].quantile(pellet_quantile)
    return pellet_interval_thres, collect_time_thres


def find_meals(data: pd.DataFrame, pellet_count_threshold=5, collect_quantile=0.6,
               pellet_quantile=0.75, verbose=False) -> list:
    """
    find meals in the behaviors. 5 pellets in 10 minutes is considered as a meal
    """
    meal_list = []
    data = data[data['Event'] == 'Pellet'].reset_index(drop=True)
    pellet_thres, collect_thres = meal_threshold(data, collect_quantile, pellet_quantile)
    window_duration = timedelta(minutes=pellet_count_threshold*pellet_thres)
    collect_threshold = pellet_count_threshold*collect_thres
    # print(window_duration, collect_threshold)
    start_idx = 0
    if verbose:
        print(f'Pellet Window Threshold: {window_duration.total_seconds()/60}min, Retrieval Window: {collect_threshold}min')

    for idx, row in data.iterrows():
        meal_start = data.iloc[start_idx]['Time']
        time_diff = row['Time'] - meal_start

        if ((row['Pellet_Count'] - data.loc[start_idx]['Pellet_Count'] >= pellet_count_threshold) and
            (time_diff <= window_duration) and
            (sum(data['collect_time'][start_idx:idx+1]) <= collect_threshold)):
            # print(data['collect_time'][start_idx:idx+1].tolist(), sum(data['collect_time'][start_idx:idx+1]))
            meal_list.append([meal_start, row['Time']])
            start_idx = idx
        elif time_diff > window_duration:
            start_idx = idx

    return meal_list


def find_meals_paper(df, time_threshold=130):
    df = df[df['Event'] == 'Pellet'].reset_index(drop=True)
    df['Time'] = pd.to_datetime(df['Time']) 
    meals = []
    meal_start_time = None
    meal_end_time = None
    window_duration = timedelta(minutes=15)
    
    for index, row in df.iterrows():
        current_time = row['Time']  # Get current time from the 'Time' column
        collect_time = row['collect_time'] * 60  # Convert collect_time to seconds
        
        if meal_start_time is None:
            meal_start_time = current_time
            meal_end_time = current_time
        # if current pellet is retrieved within 130 seconds
        if collect_time <= time_threshold and current_time - meal_start_time <= window_duration:
            meal_end_time = current_time # extend meal end time
        else:
            if meal_start_time != meal_end_time:
                meals.append([meal_start_time, meal_end_time])
            meal_start_time = current_time
            meal_end_time = current_time
    
    if meal_start_time is not None:
        meals.append([meal_start_time, meal_end_time])
    
    return meals

def graphing_cum_count(data: pd.DataFrame, meal: list, bhv, num, flip=False):
    """
    graph the cumulative count and cumulative percentage of pellet consumption
    use two axis and mark meals on the graph
    """
    fig, ax1 = plt.subplots()
    ax1.plot(data['Time'], data['Pellet_Count'], color='blue')
    if bhv == None:
        ax1.set_title(f'Pellet Count and Cumulative Sum Over Time of Mouse {num}', fontsize=18)
    else:
        ax1.set_title(f'Pellet Count and Cumulative Sum Over Time of Group {bhv} Mouse {num}', fontsize=18)

    for interval in meal:
        plt.axvspan(interval[0], interval[1], color='lightblue')

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Pellet_Count', fontsize=12)

    if not flip:
        ax2 = ax1.twinx()  # Share the same x-axis as ax1
        ax2.set_ylabel('Cum_Sum', fontsize=12)
        ax2.plot(data['Time'], data['Cum_Sum'], color='blue')

    start = None
    end = None
    
    if bhv is not None:
        for interval in pd.date_range(start=data['Time'].min(), end=data['Time'].max(), freq='20min'):
            if (19 <= interval.hour or interval.hour < 7) and start == None:
                start = interval
            elif interval.hour == 7:
                end = interval
                plt.axvspan(start, end, color='grey', alpha=0.4)
                start = end = None
        if start != None and end == None:
            plt.axvspan(start, data['Time'].max(), color='grey', alpha=0.4)
    else:
        for interval in pd.date_range(start=data['Time'].min(), end=data['Time'].max(), freq='20min'):
            if (7 <= interval.hour and interval.hour < 19) and start == None:
                start = interval
            elif interval.hour == 19 and start != None:        
                # print(start, interval)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                plt.axvspan(start, interval, color='grey', alpha=0.4)
                start = end = None
        if start != None and end == None:
            plt.axvspan(start, data['Time'].max(), color='grey', alpha=0.4)
    
    patch_meal = mpatches.Patch(color='lightblue', alpha=0.9, label='Meal')
    patch_night = mpatches.Patch(color='grey', alpha=0.5, label='Inactive')

    plt.legend(handles=[patch_meal, patch_night], loc='upper right')
    plt.show()


def experiment_duration(data: pd.DataFrame):
    """Return the duration of the experiment in unit of days

    Args:
        data (pd.DataFrame): behavior data

    Returns:
        float: durations
    """
    data['Time'] = pd.to_datetime(data['Time'])
    duration = data.tail(1)['Time'].values[0] - data.head(1)['Time'].values[0]
    duration_seconds = duration / np.timedelta64(1, 's')
    duration = duration_seconds / (60 * 60 * 24)
    return duration


def calculate_deviation(grouped_data: pd.DataFrame) -> float:
    frequency = grouped_data['Cum_Sum'].tolist()
    avg = np.median(frequency)
    deviation = [(each - avg)**2 for each in frequency]
    return sum(deviation) / len(frequency)


def active_meal(meals: list) -> float:
    """Find meals in the active period of mice (7pm-7am)
    If the start of the meal is in the active period, we count
    it as one

    Args:
        meals (list): list of meals time intervals

    Returns:
        float: percentage of meals in the active state
    """
    if len(meals) == 0:
        return 0
    cnt = 0
    for meal in meals:
        if meal[0].hour >= 19 or meal[0].hour < 7:
            cnt += 1
    return round(cnt/len(meals), 4) 

def graph_group_stats(ctrl:list, exp:list, stats_name:str, bar_width=0.2,
                      err_width=14, dpi=100, exp_name=None, verbose=True):
    """Plot bar graphs of average pellet for control and experiment groups

    Args:
        ctrl_pellet_avg (list): control data
        exp_pellet_avg (list): experiment data
        stats_name (str): the name of statistic you are graphing
        exp_name (_type_, optional): Name of the experiment group. Defaults to None.
        bar_width (float, optional): bar width of bar plot. Defaults to 0.2.
        err_width (int, optional): error bar width onthe bar. Defaults to 12.
        dpi (int, optional): dot per inch, higher dpi gives images with higher resolution. Defaults to 100.
        verbose (bool, optional): whether printing out information used in plotting. Defaults to False.
    """
    ctrl_averages = np.mean(ctrl)
    exp_averages = np.mean(exp)
    ctrl_std = np.std(ctrl, ddof=1)
    exp_std = np.std(exp, ddof=1)
    
    exp_name = 'Experiment' if exp_name == None else exp_name
    
    if verbose:
        print(f'Control Size: {len(ctrl)}')
        print(f'{exp_name} Size: {len(exp)}')
        print(f'Control Average: {ctrl_averages}')
        print(f'{exp_name} Average: {exp_averages}')
        print(f'Control Standard Deviation: {ctrl_std}')
        print(f'{exp_name} Standard Deviation: {exp_std}')

    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(6, 6)
    x = [0.5, 1]
    
    ax.bar(x=x[0], height=ctrl_averages, width=bar_width, color='blue', label='Control',
           zorder=1, alpha=0.6, yerr=ctrl_std, capsize=err_width)
    
    x_values = np.full(len(ctrl), x[0])
    ax.scatter(x_values, ctrl, marker='o', zorder=2, color='#1405eb')
    
    ax.bar(x=x[1], height=exp_averages, width=bar_width, color='orange', label=exp_name,
           zorder=1, alpha=0.6, yerr=exp_std, capsize=err_width)
    x_values = np.full(len(exp), x[1])
    ax.scatter(x_values, exp, marker='o', zorder=2, color='#f28211')

    ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylabel('Averages', fontsize=14)
    ax.set_title(f'{stats_name} of Control and {exp_name} Groups', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Control', exp_name])

    ax.legend()
    plt.show()
    