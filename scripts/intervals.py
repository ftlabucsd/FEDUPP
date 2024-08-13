import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tools import get_bhv_num
from preprocessing import get_retrieval_time
import numpy as np


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
    data = data[['MM:DD:YYYY hh:mm:ss', 'Event']].rename(columns={'MM:DD:YYYY hh:mm:ss' : 'Time'})
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
    if len(info) == 2:
        plt.title(f'Interval Between Pellets for Group {info[0]} Mouse {info[1]}', fontsize=18)
    else:
        plt.title(f'Interval Between Pellets for Mouse {info[0]}', fontsize=18)

    plt.xlabel('Time')
    plt.ylabel('Interval (minutes)')
    plt.show()
    
    
def mean_pellet_collect_time(path:str, remove_outlier=False, n_stds=3):
    pellet_times = get_retrieval_time(path)
    mean = np.mean(pellet_times)
    std = np.std(pellet_times)
    if remove_outlier:
        cutoff = mean+std*n_stds
        pellet_times = [each for each in pellet_times if each < cutoff]
    return pellet_times, np.mean(pellet_times), np.std(pellet_times)
    

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


def MannWhitneyUTest(ctrl, exp, test_side='two-sided', alpha=0.05):
    """Perform Mann-Whitney U rank test on control and experiment groups

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
    

    _, p_value = stats.mannwhitneyu(exp, ctrl, alternative=test_side)

    print("P Value is ", p_value)
    if p_value < alpha:
        if test_side == 'two-sided':
            print("There is a significant difference between the two groups.")
        else:
            print(f'Experiment group is significantly {test_side} than control group')
    else:
        print("There is no significant difference between the two groups.")

