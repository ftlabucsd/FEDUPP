"""
This script provides functions for analyzing meal patterns in FED3 data.
It includes methods for identifying meals, calculating meal-related statistics,
and visualizing meal data.
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from datetime import timedelta
import numpy as np
from scripts.accuracy import find_inactive_index, calculate_accuracy
from scripts.meal_classifiers import predict, RNNClassifier, CNNClassifier
import torch
from scripts.preprocessing import SessionData
import os

plt.rcParams['figure.figsize'] = (20, 6)

def pad_meal(each:list):
    """Pads a list with -1 until it reaches a length of 4.

    Args:
        each (list): The list to pad.

    Returns:
        list: The padded list.
    """
    size = len(each)
    while size < 4:
        each.append(-1)
        size += 1
    return each


def pellet_flip(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregates pellet events into 10-minute intervals and counts the
    number of pellets in each interval.

    Args:
        data (pd.DataFrame): A DataFrame containing FED3 data, including a 'Time'
                             column and an 'Event' column.

    Returns:
        pd.DataFrame: A DataFrame with 'Interval_Start' and 'Pellet_Count' columns.
    """
    data = data.set_index('Time')
    grouped_data = data[data['Event'] == 'Pellet'].resample('10min').size().reset_index()
    grouped_data.columns = ['Interval_Start', 'Pellet_Count']
    
    return grouped_data


def average_pellet(group: pd.DataFrame) -> float:
    """Calculates the average number of pellets consumed per 24-hour period.

    Args:
        group (pd.DataFrame): A DataFrame with 'Interval_Start' and 'Pellet_Count'
                              columns, as returned by pellet_flip().

    Returns:
        float: The average number of pellets per day.
    """
    total_hr = (group['Interval_Start'].max()-group['Interval_Start'].min()).total_seconds() / 3600
    total_pellet = group['Pellet_Count'].sum()
    return round(24*total_pellet / total_hr, 2)


def graph_pellet_frequency(grouped_data: pd.DataFrame, bhv, num, export_path=None, show: bool = False):
    """Generates a bar plot showing the frequency of pellet consumption over time."""
    fig, ax = plt.subplots()
    sns.barplot(data=grouped_data, x='Interval_Start', y='Pellet_Count', color='#000099', alpha=0.5, ax=ax)

    xtick_positions = ax.get_xticks()
    hourly_labels = [label.strftime('%H:%M') for label in grouped_data['Interval_Start'] if label.minute == 0]
    hourly_positions = [pos for pos, label in zip(xtick_positions, grouped_data['Interval_Start']) if label.minute == 0]

    ax.set_xticks(hourly_positions)
    ax.set_xticklabels(hourly_labels, rotation=45, horizontalalignment='right')

    if bhv is not None:
        inactive = find_inactive_index(hourly_labels, rev=False)
    else:
        inactive = find_inactive_index(hourly_labels, rev=True)

    for idx, each in enumerate(inactive):
        label = 'Inactive' if idx == 0 else None
        ax.axvspan(6*each[0], 6*(1+each[1]), color='grey', alpha=0.4, label=label)

    ax.axhline(y=5, color='red', linestyle='--', label='meal')
    if bhv is None:
        ax.set_title(f'Pellet Frequency of Mouse {num}', fontsize=18)
    else:
        ax.set_title(f'Pellet Frequency of Group {bhv} Mouse {num}', fontsize=18)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Number of Pellet', fontsize=14)
    ax.set_yticks(range(0, 19, 2))
    ax.legend()
    fig.tight_layout()
    if export_path:
        fig.savefig(export_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def find_first_good_meal(data:pd.DataFrame, time_threshold, pellet_threshold, model_type='cnn'):
    """Identifies the first "good" meal in a session based on a classification model.

    Args:
        data (pd.DataFrame): The raw behavior data.
        time_threshold (int): The maximum time in seconds between pellets to be considered part of the same meal.
        pellet_threshold (int): The minimum number of pellets required to form a meal.
        model_type (str, optional): The type of model to use ('lstm' or 'cnn'). Defaults to 'cnn'.

    Returns:
        tuple: A tuple containing:
            - list: A list of all identified meals with their accuracies.
            - datetime or None: The timestamp of the first good meal, or None if no good meal is found.
    """
    df = data[data['Event'] == 'Pellet'].copy()
    df['retrieval_timestamp'] = df['Time'] + pd.to_timedelta(df['collect_time'], unit='m')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lstm_ckpt = os.path.join(base_dir, 'data', 'LSTM_from_CASK.pth')
    cnn_ckpt = os.path.join(base_dir, 'data', 'CNN_from_CASK.pth')
    if model_type == 'lstm': 
        model = RNNClassifier(input_size=1, hidden_size=400, num_layers=2, num_classes=2).to(device)
        model.load_state_dict(torch.load(lstm_ckpt, map_location='cpu'))
    elif model_type == 'cnn':
        model = CNNClassifier(num_classes=2, maxlen=4).to(device)
        model.load_state_dict(torch.load(cnn_ckpt, map_location='cpu'))
    else:
        print('Only support lstm and cnn.')
        return
    
    meals_with_acc = []
    meal_start_time = None
    meal_start_index = None
    first_good_meal_index = None

    pellet_cnt = 1 # record pellets in the meal
    for index, row in df.iterrows():
        current_time = row['retrieval_timestamp']  # Get current time from the 'Time' column

        if meal_start_time is None:
            meal_start_time = current_time
            meal_start_index = index
            continue

        # if current pellet is retrieved within 60 seconds after previous retrieval
        if ((row['retrieval_timestamp'] - meal_start_time).total_seconds() <= time_threshold):
            pellet_cnt += 1
        else:
            meal_events = data.loc[meal_start_index:index]
            if pellet_cnt >= pellet_threshold and calculate_accuracy(meal_events) > 50:
                pellet_cnt += 1
                accuracies = extract_meal_acc_each(meal_events)
                meals_with_acc.append([meal_start_time, pad_meal(accuracies)])

            meal_start_time = current_time
            meal_start_index = index
            pellet_cnt = 1

    if pellet_cnt >= pellet_threshold:
        accuracies = extract_meal_acc_each(data.loc[meal_start_index:])
        meals_with_acc.append([meal_start_time, pad_meal(accuracies)])

    if len(meals_with_acc) == 0: return meals_with_acc, None # half session time if no meal

    temp_list = [row[1] for row in meals_with_acc if len(row[1]) in [2,3,4]]
    idx = -1
    if len(temp_list) == 0: # no model-recognizable meal
        for each in meals_with_acc: # search meal with average accuracy of at least 75%
            if np.mean(each[1]) >= 80:
                first_good_meal_index = idx+1
                break
            idx += 1
        # if no high-accuracy meal, then use max time
        if first_good_meal_index == None: return 0, None
    else:
        meal_data = np.stack(temp_list)
        predicted = predict(model, meal_data)
        indices = np.where(predicted == 0)[0]
        if indices.size > 0:
            first_good_meal_index = np.where(predicted==0)[0][0]
        else:
            return 0, None
    return meals_with_acc, pd.to_datetime(meals_with_acc[first_good_meal_index][0])

def extract_meal_acc_each(events: pd.DataFrame):
    """Calculates the accuracy of pellet retrieval for each inter-pellet interval within a meal.

    Args:
        events (pd.DataFrame): A DataFrame slice representing the events within a single meal.

    Returns:
        list: A list of accuracy values for each interval between pellets in the meal.
    """
    acc = []
    pellet_indices = events.index[events['Event'] == 'Pellet'].tolist()

    for idx in range(len(pellet_indices) - 1):
        start, end = pellet_indices[idx], pellet_indices[idx+1]
        curr_slice = events.loc[start:end]
        # curr_slice = curr_slice[curr_slice['Event'] != 'Pellet']
        acc.append(calculate_accuracy(curr_slice))

    # print(f"There are {len(pellet_indices)} pellets and {len(acc)} accuracy")
    return acc


def extract_meals_data(data: pd.DataFrame, time_threshold=60, 
                       pellet_threshold=2, verbose=False) -> list:
    """Extracts all meals from the data and groups them by the number of pellets.

    Args:
        data (pd.DataFrame): The raw behavior data.
        time_threshold (int, optional): The maximum time between pellets for a meal. Defaults to 60.
        pellet_threshold (int, optional): The minimum number of pellets for a meal. Defaults to 2.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        defaultdict: A dictionary where keys are the number of pellets in a meal and
                     values are lists of accuracy sequences for each meal.
    """
    df = data[data['Event'] == 'Pellet'].copy()
    df['retrieval_timestamp'] = df['Time'] + pd.to_timedelta(df['collect_time'], unit='m')

    meal_acc = defaultdict(list) # key is number of pellets, value is each series of accuracy in the meal
    meal_start_time = None
    meal_start_index = None

    pellet_cnt = 1 # record pellets in the meal
    for index, row in df.iterrows():
        current_time = row['retrieval_timestamp']  # Get current time from the 'Time' column

        if meal_start_time is None:
            meal_start_time = current_time
            meal_start_index = index
            continue

        # if current pellet is retrieved within 60 seconds after previous retrieval
        if ((row['retrieval_timestamp'] - meal_start_time).total_seconds() <= time_threshold):
            pellet_cnt += 1
        else:
            if pellet_cnt >= pellet_threshold:
                pellet_cnt += 1
                meal_events = data.loc[meal_start_index:index]
                accuracies = extract_meal_acc_each(meal_events)
                meal_acc[pellet_cnt].append(accuracies)

            meal_start_time = current_time
            meal_start_index = index
            pellet_cnt = 1

    if pellet_cnt >= pellet_threshold:
        accuracies = extract_meal_acc_each(data.loc[meal_start_index:])
        meal_acc[len(accuracies)+1].append(accuracies)
    
    return meal_acc


def find_meals_paper(data:pd.DataFrame, time_threshold=60, pellet_threshold=2, in_meal_ratio=False):
    """Identifies meals based on the criteria defined in a research paper.

    Args:
        data (pd.DataFrame): The raw behavior data.
        time_threshold (int, optional): The maximum time between pellets for a meal. Defaults to 60.
        pellet_threshold (int, optional): The minimum number of pellets for a meal. Defaults to 2.
        in_meal_ratio (bool, optional): If True, also returns the ratio of pellets
                                     consumed within meals. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - list: A list of meal time intervals.
            - list: A list of accuracies for each meal.
            - float (optional): The ratio of pellets in meals to total pellets.
    """
    df = data[data['Event'] == 'Pellet'].copy()
    df['retrieval_timestamp'] = df['Time'] + pd.to_timedelta(df['collect_time'], unit='m')

    total_pellets = len(df)            # denominator for the optional ratio
    pellets_in_meals = 0               # numerator we will accumulate

    meals, meal_acc = [], []

    meal_start_time = meal_end_time = None
    meal_start_idx = None
    pellet_cnt = 0

    for idx, row in df.iterrows():
        current_time = row['retrieval_timestamp']

        # First pellet (or first after closing a meal) → open new meal window
        if meal_start_time is None:
            meal_start_time = meal_end_time = current_time
            meal_start_idx = idx
            pellet_cnt = 1
            continue

        # Same meal if retrieved within threshold of previous pellet
        if (current_time - meal_end_time).total_seconds() <= time_threshold:
            meal_end_time = current_time
            pellet_cnt += 1
        else:
            # Close previous burst and decide whether it’s an accepted meal
            burst_events = data.loc[meal_start_idx : idx - 1]  # inclusive slice
            if pellet_cnt >= pellet_threshold and calculate_accuracy(burst_events) > 50:
                meals.append([meal_start_time, meal_end_time])
                meal_acc.append(calculate_accuracy(burst_events))
                pellets_in_meals += pellet_cnt

            # Start a new burst
            meal_start_time = meal_end_time = current_time
            meal_start_idx = idx
            pellet_cnt = 1

    burst_events = data.loc[meal_start_idx:]
    if pellet_cnt >= pellet_threshold and calculate_accuracy(burst_events) > 50:
        meals.append([meal_start_time, meal_end_time])
        meal_acc.append(calculate_accuracy(burst_events))
        pellets_in_meals += pellet_cnt

    if in_meal_ratio:
        meal_ratio = pellets_in_meals / total_pellets if total_pellets else 0
        return meals, meal_acc, meal_ratio
    return meals, meal_acc

def extract_meals_for_model(data: pd.DataFrame, time_threshold=60, 
                       pellet_threshold=2) -> list:
    """Extracts meal data formatted for input to a machine learning model.

    This function identifies meals based on the provided time and pellet thresholds,
    pads the meal data to a uniform length, and returns the meals and their lengths.

    Args:
        data (pd.DataFrame): The raw behavior data.
        time_threshold (int, optional): The maximum time between pellets in a meal. Defaults to 60.
        pellet_threshold (int, optional): The minimum number of pellets in a meal. Defaults to 2.

    Returns:
        tuple: A tuple containing:
            - list: A list of padded meal accuracy sequences.
            - list: A list of the original lengths of the meals.
    """
    df = data[data['Event'] == 'Pellet'].copy()
    df['retrieval_timestamp'] = df['Time'] + pd.to_timedelta(df['collect_time'], unit='m')

    meal_acc = []
    meal_start_time = None
    meal_start_index = None

    pellet_cnt = 1 # record pellets in the meal
    for index, row in df.iterrows():
        current_time = row['retrieval_timestamp']  # Get current time from the 'Time' column

        if meal_start_time is None:
            meal_start_time = current_time
            meal_start_index = index
            continue

        # if current pellet is retrieved within 60 seconds after previous retrieval
        if ((row['retrieval_timestamp'] - meal_start_time).total_seconds() <= time_threshold):
            pellet_cnt += 1
        else:
            if pellet_cnt >= pellet_threshold and pellet_cnt:
                pellet_cnt += 1
                meal_events = data.loc[meal_start_index:index]
                accuracies = extract_meal_acc_each(meal_events)
                meal_acc.append(accuracies)

            meal_start_time = current_time
            meal_start_index = index
            pellet_cnt = 1

    if pellet_cnt >= pellet_threshold:
        accuracies = extract_meal_acc_each(data.loc[meal_start_index:])
        meal_acc.append(accuracies)
    
    temp_list = [row for row in meal_acc if len(row) in [2,3,4]]
    meal_len = [len(each)+1 for each in temp_list]
    meals = [pad_meal(meal) for meal in temp_list]
    return meals, meal_len


def graphing_cum_count(data: pd.DataFrame, meal: list, bhv, num, flip=False, export_path=None, show: bool = False):
    """Graphs the cumulative count of pellets over time."""
    fig, ax1 = plt.subplots()
    ax1.plot(data['Time'], data['Pellet_Count'], color='blue')
    if bhv is None:
        ax1.set_title(f'Pellet Count and Cumulative Sum Over Time of Mouse {num}', fontsize=18)
    else:
        ax1.set_title(f'Pellet Count and Cumulative Sum Over Time of Group {bhv} Mouse {num}', fontsize=18)

    for interval in meal:
        ax1.axvspan(interval[0], interval[1], color='lightblue')

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Pellet_Count', fontsize=12)

    if not flip:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Cum_Sum', fontsize=12)
        ax2.plot(data['Time'], data['Cum_Sum'], color='blue')

    start = None
    end = None

    if bhv is not None:
        for interval in pd.date_range(start=data['Time'].min(), end=data['Time'].max(), freq='20min'):
            if (19 <= interval.hour or interval.hour < 7) and start is None:
                start = interval
            elif interval.hour == 7 and start is not None:
                ax1.axvspan(start, interval, color='grey', alpha=0.4)
                start = end = None
        if start is not None and end is None:
            ax1.axvspan(start, data['Time'].max(), color='grey', alpha=0.4)
    else:
        for interval in pd.date_range(start=data['Time'].min(), end=data['Time'].max(), freq='20min'):
            if (7 <= interval.hour < 19) and start is None:
                start = interval
            elif interval.hour == 19 and start is not None:
                ax1.axvspan(start, interval, color='grey', alpha=0.4)
                start = end = None
        if start is not None and end is None:
            ax1.axvspan(start, data['Time'].max(), color='grey', alpha=0.4)

    patch_meal = mpatches.Patch(color='lightblue', alpha=0.9, label='Meal')
    patch_inactive = mpatches.Patch(color='grey', alpha=0.5, label='Inactive')

    ax1.legend(handles=[patch_meal, patch_inactive], loc='upper right')
    if export_path:
        fig.savefig(export_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def experiment_duration(data: pd.DataFrame):
    """Calculates the total duration of the experiment in days.

    Args:
        data (pd.DataFrame): The behavior data with a 'Time' column.

    Returns:
        float: The duration of the experiment in days.
    """
    data['Time'] = pd.to_datetime(data['Time'])
    duration = data.tail(1)['Time'].values[0] - data.head(1)['Time'].values[0]
    duration_seconds = duration / np.timedelta64(1, 's')
    duration = duration_seconds / (60 * 60 * 24)
    return duration


def calculate_deviation(grouped_data: pd.DataFrame) -> float:
    """Calculates the mean squared deviation of cumulative pellet counts.

    Args:
        grouped_data (pd.DataFrame): DataFrame with a 'Cum_Sum' column.

    Returns:
        float: The mean squared deviation.
    """
    frequency = grouped_data['Cum_Sum'].tolist()
    avg = np.median(frequency)
    deviation = [(each - avg)**2 for each in frequency]
    return sum(deviation) / len(frequency)


def active_meal(meals: list) -> float:
    """Calculates the percentage of meals that occur during the active period (7pm-7am).

    A meal is considered active if its start time falls within the active period.

    Args:
        meals (list): A list of meal time intervals, where each interval is a list
                      of [start_time, end_time].

    Returns:
        float: The percentage of meals that occurred during the active period.
    """
    if len(meals) == 0:
        return 0
    cnt = 0
    for meal in meals:
        if meal[0].hour >= 19 or meal[0].hour < 7:
            cnt += 1
    return round(cnt/len(meals), 4) 


def print_meal_stats(data):
    """Prints statistics about the number of meals of different sizes.

    Args:
        data (dict): A dictionary where keys are the number of pellets in a meal
                     and values are lists of meals.
    """
    total_meals = 0
    keep_meals = 0
    for key, item in data.items():
        size = len(item)
        total_meals += size
        if key in [3, 4, 5]:
            keep_meals += size
            print(f"Number of Pellets: {key}, n_meals: {size}")
    print(f"Total {total_meals} meals and keep {keep_meals}")


def process_meal_data(session: SessionData, export_root: str | os.PathLike | None = None, prefix: str | None = None):
    """Processes meal data for a single sheet and returns key metrics.

    This function reads data from a single Excel sheet, identifies meals, calculates
    various meal-related metrics, and optionally generates and saves plots.

    Args:
        sheet (str): The name of the Excel sheet to process.
        path (str): The path to the Excel file.
        is_cask (bool, optional): Flag indicating if the data is from a CASK mouse. Defaults to False.
        export_root (str, optional): The root directory to save generated graphs. Defaults to None.
        prefix (str, optional): The prefix for the output file names. Defaults to None.

    Returns:
        dict: A dictionary containing calculated meal metrics.
    """
    data = session.raw.copy()
    meal, _, in_meal_ratio = find_meals_paper(data, time_threshold=60, pellet_threshold=2, in_meal_ratio=True)
    meal_with_acc, first_meal_time = find_first_good_meal(data, 60, 2, 'lstm')
    meal_1 = (meal[0][0] - data['Time'][0]).total_seconds() / 3600 if meal else 0
    meal_1_good = (
        (first_meal_time - data['Time'][0]).total_seconds() / 3600
        if first_meal_time is not None
        else meal_1
    )
    group = pellet_flip(data)
    bhv, num = session.key.group, session.key.mouse_id
    
    run_prefix = prefix if prefix is not None else session.key.group.lower()
    
    session_label = session.key.session_id
    export_root_path = Path(export_root) if export_root else None
    freq_path = (
        export_root_path / f"{run_prefix}_{session_label}_pellet_frequency.svg"
        if export_root_path
        else None
    )
    cum_path = (
        export_root_path / f"{run_prefix}_{session_label}_cumulative_sum.svg"
        if export_root_path
        else None
    )
    
    graph_pellet_frequency(
        group,
        bhv,
        num,
        export_path=str(freq_path) if freq_path else None,
        show=False,
    )
    graphing_cum_count(
        data,
        meal,
        bhv,
        num,
        flip=True,
        export_path=str(cum_path) if cum_path else None,
        show=False,
    )
    
    return {
        'avg_pellet': average_pellet(group),
        'inactive_meals': active_meal(meal),
        'fir_meal': meal_1,
        'fir_good_meal': meal_1_good,
        'meal_count': round(len(meal) / experiment_duration(data), 2),
        'in_meal_ratio': in_meal_ratio
    }