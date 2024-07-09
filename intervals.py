import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tools import get_bhv_num


def count_interval(data: pd.DataFrame) -> list:
    intervals = []
    
    for i in range(1, len(data)):
        current_timestamp = data.iloc[i]['Time']
        previous_timestamp = data.iloc[i - 1]['Time']
        
        interval = (current_timestamp - previous_timestamp).total_seconds() / 60
        intervals.append(interval)
    
    return intervals


def clean_and_interval(data: pd.DataFrame) -> pd.DataFrame:
    """Add interval column to the data

    Interval column means the time interval between current action and previous action
    in terms of minutes
    
    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    data = data[['MM:DD:YYYY hh:mm:ss', 'Event']].rename(columns={'MM:DD:YYYY hh:mm:ss' : 'Time'})
    data = data[data['Event'] == 'Pellet'].reset_index().drop('index', axis='columns')
    data['Time'] = pd.to_datetime(data['Time'])
    data['Interval'] = data['Time'].diff().fillna(pd.Timedelta(seconds=0))
    data['Interval'] = data['Interval'].dt.total_seconds() / 60
    return data


# def get_bhv_num(path: str) -> tuple:
#     branches = path.split(sep='/')
    
#     num = branches[2][1]
#     bhv = branches[1][4]

#     return bhv, num


def graph_pellet_interval(path: str):
    data = pd.read_csv(path)
    data = clean_and_interval(data)
    
    info = get_bhv_num(path)
    plt.figure(figsize=(15, 5))

    sns.set_palette('bright')
    sns.set_style('darkgrid')

    sns.lineplot(data=data, x='Time', y='Interval', alpha=0.8)

    if len(info) == 2:
        plt.title(f'Interval Between Pellets for Group {info[0]} Mouse {info[1]}', fontsize=18)
    else:
        plt.title(f'Interval Between Pellets for Mouse {info}', fontsize=18)

    plt.xlabel('Time')
    plt.ylabel('Interval (minutes)')
    plt.show()
    

def perform_T_test(ctrl:list, exp:list, test_side:str, alpha=0.05, paired=False):
    """Perform T tests on control and experiment groups

    Args:
        ctrl (list): data from control group
        exp (list): data from experiment group
        test_side (str): the alternative hypothesis 
                two-sided -> not equal
                greater -> exp mean > ctrl mean
                less -> exp mean < ctrl mean
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


def MannWhitneyUTest(ctrl, exp, test_side:str, alpha=0.05):
    """Perform T tests on control and experiment groups

    Args:
        ctrl (list): data from control group
        exp (list): data from experiment group
        test_side (str): the alternative hypothesis 
                two-sided -> not equal
                greater -> exp mean > ctrl mean
                less -> exp mean < ctrl mean
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

