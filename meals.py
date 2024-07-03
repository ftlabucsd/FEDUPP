import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from datetime import timedelta
import numpy as np
from accuracy import find_night_index

plt.rcParams['figure.figsize'] = (20, 6)


def pellet_flip(data: pd.DataFrame) -> pd.DataFrame:
    data = data.set_index('Time')
    grouped_data = data[data['Event'] == 'Pellet'].resample('10T').size().reset_index()
    grouped_data.columns = ['Interval_Start', 'Pellet_Count']
    
    return grouped_data


def average_pellet(group: pd.DataFrame) -> float:
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
    dark = find_night_index(hourly_labels)

    for idx, each in enumerate(dark):
        if idx == 0:
            ax.axvspan(6*each[0], 6*(1+each[1]), color='grey', alpha=0.4, label='Night')
        else:
            ax.axvspan(6*each[0], 6*(1+each[1]), color='grey', alpha=0.4)

    # Add vertical grey background for the time interval between 7 p.m. and 7 a.m.
    plt.axhline(y=5, color='red', linestyle='--', label='meal')
    if bhv == None or num == None:
        plt.title(f'Pellet Frequency', fontsize=18)
    else:
        plt.title(f'Pellet Frequency of Group {bhv} Mice {num}', fontsize=18)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Number of Pellet', fontsize=14)
    plt.yticks(range(0, 19, 2))
    plt.tight_layout()
    plt.legend()
    plt.show()


def find_meals(data: pd.DataFrame) -> list:
    """
    find meals in the behaviors. 5 pellets in 10 minutes is considered as a meal
    """
    meal_list = []
    pellet_count_threshold = 5
    window_duration = timedelta(minutes=10)
    start_idx = 0

    for idx, row in data.iterrows():
        meal_start = data.iloc[start_idx]['Time']
        time_diff = row['Time'] - meal_start

        if (row['Pellet_Count'] - data.loc[start_idx]['Pellet_Count'] >= 
                    pellet_count_threshold) and (time_diff <= window_duration):
            
            meal_list.append([meal_start, row['Time']])
            start_idx = idx
        elif time_diff > window_duration:
            start_idx = idx

    return meal_list


def graphing_cum_count(data: pd.DataFrame, meal: list, bhv: int, num: int, flip=False):
    """graph the cumulative count and percentage of pellet consumption
    use two axis and mark meals on the graph
    """
    fig, ax1 = plt.subplots()
    ax1.plot(data['Time'], data['Cum_Sum'], color='blue')
    if bhv == None or num == None:
        ax1.set_title(f'Pellet Count and Cumulative Sum Over Time', fontsize=18)
    else:
        ax1.set_title(f'Pellet Count and Cumulative Sum Over Time of Group {bhv} Mice {num}', fontsize=18)

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
    for interval in pd.date_range(start=data['Time'].min(), end=data['Time'].max(), freq='20min'):
        if (19 <= interval.hour or interval.hour < 7) and start == None:
            start = interval
        elif interval.hour == 7:
            end = interval
            plt.axvspan(start, end, color='grey', alpha=0.4)
            start = end = None
    if start != None and end == None:
        plt.axvspan(start, data['Time'].max(), color='grey', alpha=0.4)
    
    patch_meal = mpatches.Patch(color='lightblue', alpha=0.8, label='Meal')
    patch_night = mpatches.Patch(color='grey', alpha=0.5, label='Inactive')

    plt.legend(handles=[patch_meal, patch_night], loc='upper right')
    plt.show()


def experiment_duration(data: pd.DataFrame):
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


def inactive_meal(meals: list) -> float:
    cnt = 0
    for meal in meals:
        if meal[0].hour >= 19 or meal[0].hour < 7:
            cnt += 1
    return round(cnt/len(meals), 4) 


def graph_average_pellet(ctrl_pellet_avg:list, exp_pellet_avg:list, exp_name=None):
    exp_name = 'Experiment' if exp_name == None else exp_name
    
    # Create DataFrames for each group
    data_ctrl = pd.DataFrame({'Group': 'Control', 'Value': ctrl_pellet_avg})
    data_cask = pd.DataFrame({'Group': exp_name, 'Value': exp_pellet_avg})

    # Concatenate the two DataFrames
    data = pd.concat([data_ctrl, data_cask])

    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")

    # Create the bar plot with error bars
    sns.barplot(x="Group", y="Value", data=data, palette="pastel",
                    errorbar="sd", capsize=0.2, width=0.5, errcolor='0.4')

    plt.title('Average Pellets Per Hour for FR1', fontsize=16)
    plt.xlabel('Groups')
    plt.ylabel('Average Pellets')
    plt.show()

    