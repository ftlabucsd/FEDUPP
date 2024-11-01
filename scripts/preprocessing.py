import pandas as pd
import math
import numpy as np
import datetime

def read_excel_by_sheet(sheet, parent='../behavior data integrated/Adjusted FED3 Data.xlsx', 
                        hundredize=True, convert_time=True, remove_trival=True):
    """
    Read excel file with certain sheet name. Replace all RightWithPellet and LeftWithPellet to 
    Right and left. It will automatically convert accuracy to 0-100 scale and remove data at the
    beginning if the percent correct is 0 (start with first non-zero data).

    Parameters:
    parent: excel file path
    sheet: sheet name
    hundredize: whether converting accuracy from 0-1 to 0-100
    convert_time: whether converting time column to datetime format for processing
    remove_trivial: whether removing no-pellet region at the beginning. The first entry would
        become first pellet behavior
    """
    df = pd.read_excel(parent, sheet_name=sheet)

    df = df[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke', 'Pellet_Count',
         'Cum_Sum', 'Percent_Correct']].rename(columns={'MM:DD:YYYY hh:mm:ss': 'Time'}).dropna()
    
    df = df.replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                    'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})
    baseline_time = df['Time'].iloc[0]
    df['Time_passed'] = df['Time'] - baseline_time
    
    if hundredize:
        df['Percent_Correct'] *= 100
    if convert_time:
        df['Time'] = pd.to_datetime(df['Time'])
    if remove_trival:
        mask = (df['Percent_Correct'] != 0) & (df['Cum_Sum'] != 0)
        df = df[mask]

    df = df.reset_index().drop(['index'], axis='columns')
    return df


def read_csv_clean(path:str, remove_trivial=True, cumulative_accuracy=False, 
                   convert_large=False, collect_time=False) -> pd.DataFrame:
    """Read csv file and clean it

    Args:
        path (str): path of the csv file
        remove_trivial (bool, optional): remove rows until the first pellet. Defaults to True.
        cumulative_accuracy (bool, optional): whether calcualate cumulative accuracy for each row. Defaults to False.
        convert_large (bool, optional): whether convert accuracy to 0-100 scale from 0-1 scale. Defaults to False.

    Returns:
        pd.DataFrame: cleaned data
    """
    if path.startswith('.'): return None
    
    all = pd.read_csv(path)
    
    if 'Time' not in all.columns:
        df = all[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke', 'Pellet_Count', 
                 'Left_Poke_Count', 'Right_Poke_Count']].rename(columns={
                'MM:DD:YYYY hh:mm:ss': 'Time'}).dropna()
        
        df = df.replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                        'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})
    
        df['Cum_Sum'] = df['Pellet_Count'] / max(df['Pellet_Count'])
    df['Time'] = pd.to_datetime(df['Time'])
    baseline_time = df['Time'].iloc[0]
    df['Time_passed'] = df['Time'] - baseline_time
    df = df.reset_index().drop(['index'], axis='columns')
    
    if cumulative_accuracy:
        df = calculate_accuracy_by_row(df, convert_large)
    
    if collect_time:
        df['collect_time'] = all['Retrieval_Time']
        df['collect_time'] = pd.to_numeric(df['collect_time'], errors='coerce')
        max_value = df['collect_time'].max()
        # print("Replaced Timed_out with max value", max_value)
        df['collect_time'] = df['collect_time'].replace('Timed_out', max_value)
        df.loc[np.isnan(df['collect_time']), "collect_time"] = 0
    
    if remove_trivial:
        first_non_zero_index = df['Cum_Sum'].ne(0).idxmax()
        df = df.loc[first_non_zero_index:]
        df.reset_index(drop=True, inplace=True)
    
    return df


def calculate_accuracy_by_row(df:pd.DataFrame, convert_large=True):
    """calculate accuracy at each time stamp

    Args:
        df (pd.DataFrame): data from cleaned csv
        convert_large (bool): whether convert accuracy from 0-1 scale to 0-100 scale
    """
    # if df['Active_Poke'].nunique() > 1:
    #     raise RuntimeError("Cumulative Accuracy only valids for FR1 data")\
    
    active_poke = df['Active_Poke'].loc[0]
    df['Percent_Correct'] = df[f'{active_poke}_Poke_Count'] / (df['Left_Poke_Count']+df['Right_Poke_Count'])
    
    
    if convert_large:
        df['Percent_Correct'] *= 100
        
    return df
        
    
def get_retrieval_time(path:str):
    df = pd.read_csv(path)
    times = df['Retrieval_Time'].tolist()
    pellet_times = [each for each in times if each != 'Timed_out']
    pellet_times = list(map(float, pellet_times))
    pellet_times = [each for each in pellet_times if not math.isnan(each)]
    return pellet_times