import pandas as pd
from scipy import stats


def count_interval(data: pd.DataFrame) -> list:
    intervals = []
    
    for i in range(1, len(data)):
        current_timestamp = data.iloc[i]['Time']
        previous_timestamp = data.iloc[i - 1]['Time']
        
        interval = (current_timestamp - previous_timestamp).total_seconds() / 60
        intervals.append(interval)
    
    return intervals
    

def perform_T_test(listOne, listTwo, alpha=0.05):
    t_statistic, p_value = stats.ttest_ind(listOne, listTwo)

    print("P Value is ", p_value)
    if p_value < alpha:
        print("There is a significant difference between the two groups.")
    else:
        print("There is no significant difference between the two groups.")
