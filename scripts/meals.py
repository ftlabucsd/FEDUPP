from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from datetime import timedelta
import numpy as np
from accuracy import find_night_index, calculate_accuracy
from meal_classifiers import predict, RNNClassifier, CNNClassifier
import torch

plt.rcParams['figure.figsize'] = (20, 6)

def pad_meal(each:list):
    size = len(each)
    while size < 4:
        each.append(-1)
        size += 1
    return each


def get_daily_pellet_counts(df, time_column='Time', pellet_column='Pellet_Count'):
    # df = pd.read_csv(input_csv)
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column)
    
    df['date'] = df[time_column].dt.date
    daily_last = df.groupby('date')[pellet_column].last().sort_index()

    daily_counts = []
    last_values = daily_last.tolist()
    
    if last_values:
        daily_counts.append(last_values[0])
        for prev, curr in zip(last_values, last_values[1:]):
            daily_counts.append(curr - prev)
    return daily_counts

def plot_daily_pellet_counts(daily_counts_2d, group, export_path=None):
    plt.figure(figsize=(10, 6), dpi=150)
    
    for i, counts in enumerate(daily_counts_2d):
        plt.plot(counts, marker='o', label=f"Mouse {i+1}")
    
    plt.xlabel("Day Index")
    plt.ylabel("Daily Pellet Count")
    plt.title(f"Daily Pellet Count of {group} Group")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()


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
    """return average day pellet count in a session

    Args:
        group (pd.DataFrame): data processed by pellet flip

    Returns:
        float: average pellet count
    """
    total_hr = (group['Interval_Start'].max()-group['Interval_Start'].min()).total_seconds() / 3600
    total_pellet = group['Pellet_Count'].sum()
    return round(24*total_pellet / total_hr, 2)


def find_pellet_frequency(data: pd.DataFrame) -> pd.DataFrame:
    """find number of pellet in every 10 minutes
    return a new data frame records the 10 minutes pellet
    """
    data = data.drop(['Pellet_Count'], axis='columns')
    data.set_index('Time', inplace=True)
    grouped_data = data[data['Event'] == 'Pellet'].resample('10min').size().reset_index()
    grouped_data.columns = ['Interval_Start', 'Pellet_Count']
    
    return grouped_data


def graph_pellet_frequency(grouped_data: pd.DataFrame, bhv, num, export_path=None):
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
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()

def find_first_good_meal(data:pd.DataFrame, time_threshold, pellet_threshold, model_type='cnn'):
    df = data[data['Event'] == 'Pellet'].copy()
    df['retrieval_timestamp'] = df['Time'] + pd.to_timedelta(df['collect_time'], unit='m')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'lstm': 
        model = RNNClassifier(input_size=1, hidden_size=400, num_layers=2, num_classes=2).to(device)
        model.load_state_dict(torch.load('../data/LSTM_from_CASK.pth'))
    elif model_type == 'cnn':
        model = CNNClassifier(num_classes=2, maxlen=4).to(device)
        model.load_state_dict(torch.load('../data/CNN_from_CASK.pth'))
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
            if np.mean(each[1]) >= 70:
                first_good_meal_index = idx+1
                break
            idx += 1
        # if no high-accuracy meal, then use max time
        if first_good_meal_index == None: first_good_meal_index = len(meals_with_acc) - 1
    else:
        meal_data = np.stack(temp_list)
        predicted = predict(model, meal_data)
        indices = np.where(predicted == 0)[0]
        first_good_meal_index = np.where(predicted==0)[0][0] if indices.size > 0 else len(meals_with_acc) - 1

    return meals_with_acc, pd.to_datetime(meals_with_acc[first_good_meal_index][0])

def extract_meal_acc_each(events: pd.DataFrame):
    acc = []
    pellet_indices = events.index[events['Event'] == 'Pellet'].tolist()

    for idx in range(len(pellet_indices) - 1):
        start, end = pellet_indices[idx], pellet_indices[idx+1]
        curr_slice = events.loc[start:end]
        # curr_slice = curr_slice[curr_slice['Event'] != 'Pellet']
        acc.append(calculate_accuracy(curr_slice))

    # print(f"There are {len(pellet_indices)} pellets and {len(acc)} accuracy")
    return acc


def extract_meals_data(data: pd.DataFrame, time_threshold=130, 
                       pellet_threshold=3, verbose=False) -> list:
    """
    find meals in the behaviors. 5 pellets in 10 minutes is considered as a meal
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


def find_meals_paper(data:pd.DataFrame, time_threshold=130, pellet_threshold=3):
    df = data[data['Event'] == 'Pellet'].copy()
    df['retrieval_timestamp'] = df['Time'] + pd.to_timedelta(df['collect_time'], unit='m')

    meals = []
    meal_acc = []
    meal_start_time = None
    meal_end_time = None
    meal_start_index = None

    pellet_cnt = 1 # record pellets in the meal
    for index, row in df.iterrows():
        current_time = row['retrieval_timestamp']  # Get current time from the 'Time' column

        if meal_start_time is None:
            meal_start_time = current_time
            meal_end_time = current_time
            meal_start_index = index

        # if current pellet is retrieved within 130 seconds after previous retrieval
        if ((row['retrieval_timestamp'] - meal_start_time).total_seconds() <= time_threshold):
            meal_end_time = current_time # extend meal end time
            pellet_cnt += 1
        else:
            meal_events = data.loc[meal_start_index:index]
            if pellet_cnt >= pellet_threshold and calculate_accuracy(meal_events) > 50:
                meals.append([meal_start_time, meal_end_time])
                meal_acc.append(calculate_accuracy(meal_events))

            meal_start_time = current_time
            meal_end_time = current_time
            meal_start_index = index
            pellet_cnt = 1

    if pellet_cnt >= pellet_threshold:
        meals.append([meal_start_time, meal_end_time])
        meal_events = data.loc[meal_start_index:index]
        meal_acc.append(calculate_accuracy(meal_events))
    return meals, meal_acc


def extract_meals_for_model(data: pd.DataFrame, time_threshold=60, 
                       pellet_threshold=2) -> list:
    """
    find meals in the behaviors. 5 pellets in 10 minutes is considered as a meal
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


def graphing_cum_count(data: pd.DataFrame, meal: list, bhv, num, flip=False, export_path=None):
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
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
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


def print_meal_stats(data):
    total_meals = 0
    keep_meals = 0
    for key, item in data.items():
        size = len(item)
        total_meals += size
        if key in [3, 4, 5]:
            keep_meals += size
            print(f"Number of Pellets: {key}, n_meals: {size}")
    print(f"Total {total_meals} meals and keep {keep_meals}")
