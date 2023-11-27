from re import I
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from datetime import timedelta, datetime
import numpy as np

plt.rcParams['figure.figsize'] = (20, 6)

def process_csv(path: str) -> pd.DataFrame:
    """Preprocess the csv file for analysis
    This function has similar functions with process_sheet
    """
    df = pd.read_csv(path)
    df = df[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke', 'Pellet_Count']].rename(
        columns={'MM:DD:YYYY hh:mm:ss': 'Time'})
    df = df.replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                                   'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.reset_index().drop(['index'], axis='columns')
    
    first_non_zero_index = df['Pellet_Count'].ne(0).idxmax()
    df = df.loc[first_non_zero_index:]
    df.reset_index(drop=True, inplace=True)
    return df


def pellet_flip(data: pd.DataFrame) -> pd.DataFrame:
    data.set_index('Time', inplace=True)
    grouped_data = data[data['Event'] == 'Pellet'].resample('10T').size().reset_index()
    grouped_data.columns = ['Interval_Start', 'Pellet_Count']
    
    return grouped_data


def average_pellet(group: pd.DataFrame) -> float:
    total_hr = (group['Interval_Start'].max()-group['Interval_Start'].min()).total_seconds() / 3600
    total_pellet = group['Pellet_Count'].sum()
    return round(total_pellet / total_hr, 3)


def process_sheet(path: str, sheet: str) -> pd.DataFrame:
    """Preprocess the excel sheet for analysis
    keep only useful columns and rename improper named column
    drop null value and convert time to datetime format
    reset index to start with 0
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df = df[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke', 'Pellet_Count', 'Cum_Sum']].rename(
        columns={'MM:DD:YYYY hh:mm:ss': 'Time'})
    df.dropna(inplace=True)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.reset_index().drop(['index'], axis='columns')
    return df


def find_pellet_frequency(data: pd.DataFrame) -> pd.DataFrame:
    """find number of pellet in every 10 minutes
    return a new data frame records the 10 minutes pellet
    """
    data = data.drop(['Pellet_Count', 'Cum_Sum'], axis='columns')
    data.set_index('Time', inplace=True)
    grouped_data = data[data['Event'] == 'Pellet'].resample('10T').size().reset_index()
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
    dark = []
    temp = {}
    for idx, tick in enumerate(hourly_labels):
        if tick == '07:00':
            temp['morning'] = hourly_positions[idx]
        elif tick == '19:00':
            temp['evening'] = hourly_positions[idx]
        
        if len(temp) == 2:
            dark.append(temp)
            temp = {}

    # start at one time, but the end did not stop
    if len(temp) == 1:
        if 'morning' in temp.keys():
            temp['evening'] = hourly_positions[0]
        else:
            temp['morning'] = len(grouped_data)-1
        dark.append(temp)

    for idx, each in enumerate(dark):
        stamps = list(each.values())
        if idx == 0:
            ax.axvspan(stamps[0], stamps[1], color='grey', alpha=0.4, label='Night')
        else:
            ax.axvspan(stamps[0], stamps[1], color='grey', alpha=0.4)

    # Add vertical grey background for the time interval between 7 p.m. and 7 a.m.
    plt.axhline(y=5, color='red', linestyle='--', label='meal')
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
        meal_start = data.loc[start_idx, 'Time']
        time_diff = row['Time'] - meal_start

        if (row['Pellet_Count'] - data.loc[start_idx, 'Pellet_Count'] >= 
                    pellet_count_threshold) and (time_diff <= window_duration):
            meal_list.append([meal_start, row['Time']])
            start_idx = idx
        elif time_diff > window_duration:
            start_idx = idx

    return meal_list


def graphing_cum_count(data: pd.DataFrame, meal: list, bhv: int, num: int):
    """graph the cumulative count and percentage of pellet consumption
    use two axis and mark meals on the graph
    """
    fig, ax1 = plt.subplots()
    ax1.plot(data['Time'], data['Pellet_Count'], color='blue')
    ax1.set_title(f'Pellet Count and Cumulative Sum Over Time of Group {bhv} Mice {num}', fontsize=18)

    for interval in meal:
        plt.axvspan(interval[0], interval[1], color='lightblue')

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Pellet_Count', fontsize=12)
    ax2 = ax1.twinx()  # Share the same x-axis as ax1
    ax2.set_ylabel('Cum_Sum', fontsize=12)
    ax2.plot(data['Time'], data['Cum_Sum'], color='blue')

    start = None
    end = None
    for interval in pd.date_range(start=data['Time'].min(), end=data['Time'].max(), freq='20T'):
        if (19 <= interval.hour or interval.hour < 7) and start == None:
            start = interval
            print(start)
        elif interval.hour == 7:
            end = interval
            plt.axvspan(start, end, color='lightgrey', alpha=0.4)
            start = end = None
    if start != None and end == None:
        plt.axvspan(start, data['Time'].max(), color='lightgrey', alpha=0.4)
    
    patch_meal = mpatches.Patch(color='lightblue', alpha=0.8, label='Meal')
    patch_night = mpatches.Patch(color='lightgrey', alpha=0.3, label='Inactive')

    plt.legend(handles=[patch_meal, patch_night], loc='upper right')
    plt.show()

def experiment_duration(data: pd.DataFrame):
    data['Time'] = pd.to_datetime(data['Time'])
    duration = data.tail(1)['Time'].values[0] - data.head(1)['Time'].values[0]
    duration_seconds = duration / np.timedelta64(1, 's')
    duration = duration_seconds / (60 * 60 * 24)
    return duration

def calculate_deviation(grouped_data: pd.DataFrame) -> float:
    frequency = grouped_data['Pellet_Count'].tolist()
    avg = np.median(frequency)
    deviation = [(each - avg)**2 for each in frequency]
    return sum(deviation) / len(frequency)

def pellet_dark(grouped_data: pd.DataFrame) -> float:
    total = 0
    dark_pellet = 0

    for _, data in grouped_data.iterrows():  
        if data['Pellet_Count'] > 0:
            total += 1 
            if data['Interval_Start'].hour >= 19 or data['Interval_Start'].hour < 7:
                dark_pellet += 1
    
    return round(dark_pellet / total, 3)
