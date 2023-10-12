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
    df = df[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke']].rename(
        columns={'MM:DD:YYYY hh:mm:ss': 'Time'})
    df = df.replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                                   'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.reset_index().drop(['index'], axis='columns')
    return df


def pellet_flip(data: pd.DataFrame) -> pd.DataFrame:
    data.set_index('Time', inplace=True)
    grouped_data = data[data['Event'] == 'Pellet'].resample('20T').size().reset_index()
    grouped_data.columns = ['Interval_Start', 'Pellet_Count']
    
    return grouped_data


def average_pellet(group: pd.DataFrame) -> float:
    total_hr = (group['Interval_Start'].max()-group['Interval_Start'].min()).total_seconds() / 3600
    total_pellet = group['Pellet_Count'].sum()
    # print('Average pellet per hour:', total_pellet / total_hr)
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
    ax = sns.barplot(data=grouped_data, x='Interval_Start', y='Pellet_Count')

    # Get the x-axis positions
    xtick_positions = ax.get_xticks()

    # Update x-axis labels to display one-hour intervals
    hourly_labels = [label.strftime('%H:%M') for label in grouped_data['Interval_Start'] if label.minute == 0]
    hourly_positions = [pos for pos, label in zip(xtick_positions, grouped_data['Interval_Start']) if label.minute == 0]

    ax.set_xticks(hourly_positions)  # Set the tick positions to match the hourly intervals
    ax.set_xticklabels(hourly_labels, rotation=45, horizontalalignment='right')  # Set the tick labels to hourly format
    
    plt.axhline(y=5, color='red', linestyle='--', label='meal')
    plt.title(f'Pellet Frequency of Group {bhv} Mice {num}', fontsize=18)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Number of Pellet', fontsize=14)
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
        elif time_diff > window_duration:
            start_idx = idx

    return meal_list


def graphing_cum_count(data: pd.DataFrame, meal: list, bhv: int, num: int):
    """graph the cumulative count and percentage of pellet consumption
    use two axis and mark meals on the graph
    """
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Pellet_Count', fontsize=12)
    ax1.plot(data['Time'], data['Pellet_Count'], color='blue')

    ax2 = ax1.twinx()  # Share the same x-axis as ax1
    ax2.set_ylabel('Cum_Sum', fontsize=12)
    ax2.plot(data['Time'], data['Cum_Sum'], color='blue')

    ax1.set_title(f'Pellet Count and Cumulative Sum Over Time of Group {bhv} Mice {num}', fontsize=18)

    for interval in meal:
        plt.axvspan(interval[0], interval[1], color='lightblue', alpha=0.8)

    patch = mpatches.Patch(color='lightblue', alpha=0.8, label='Meal')
    plt.legend(handles=[patch], loc='upper left')
    plt.show()

def experiment_duration(data: pd.DataFrame):
    data['Time'] = pd.to_datetime(data['Time'])
    duration = data.tail(1)['Time'].values[0] - data.head(1)['Time'].values[0]
    duration_seconds = duration / np.timedelta64(1, 's')
    duration = duration_seconds / (60 * 60 * 24)
    return duration