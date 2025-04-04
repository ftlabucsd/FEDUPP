import pandas as pd
import math
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


def convert_to_numeric(value):
    if isinstance(value, str) and value.isnumeric():
        return pd.to_numeric(value)
    else:
        return value

def get_all_sheet_names(excel_path:str):
    xls = pd.ExcelFile(excel_path)
    return  xls.sheet_names


def read_excel_by_sheet(sheet, parent, convert_large=False, collect_time=True,
                        cumulative_accuracy=True, remove_trivial=False):
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
    if 'Cum_Sum' not in df.columns:
        df['Cum_Sum'] = df['Pellet_Count'] / max(df['Pellet_Count'])

    df = df[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke', 'Pellet_Count', 
            'Left_Poke_Count', 'Right_Poke_Count', 'Cum_Sum', 'Retrieval_Time']].rename(columns={
            'MM:DD:YYYY hh:mm:ss': 'Time', 'Retrieval_Time': 'collect_time'
            })

    df = df.replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                    'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.reset_index(drop=True)

    if cumulative_accuracy:
        df = calculate_accuracy_by_row(df, convert_large)

    if collect_time:
        df['collect_time'] = df['collect_time'].apply(convert_to_numeric)
        max_value = get_max_numeric(df['collect_time'].copy())
        df['collect_time'] = df['collect_time'].replace('Timed_out', max_value)
        df['collect_time'] = df['collect_time'].fillna(0)
        df['collect_time'] = pd.to_numeric(df['collect_time'])

    if remove_trivial:
        first_non_zero_index = df['Cum_Sum'].ne(0).idxmax()
        df = df.loc[first_non_zero_index:]

    df = df.reset_index(drop=True)
    baseline_time = df['Time'].iloc[0]
    df['Time_passed'] = df['Time'] - baseline_time
    return df


def preprocess_csv(path, convert_large=False, collect_time=True,
                        cumulative_accuracy=True, remove_trivial=False):
    df = pd.read_csv(path)
    if 'Cum_Sum' not in df.columns:
        df['Cum_Sum'] = df['Pellet_Count'] / max(df['Pellet_Count'])

    df = df[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke', 'Pellet_Count', 
            'Left_Poke_Count', 'Right_Poke_Count', 'Cum_Sum', 'Retrieval_Time']].rename(columns={
            'MM:DD:YYYY hh:mm:ss': 'Time', 'Retrieval_Time': 'collect_time'
            })

    df = df.replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                    'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.reset_index(drop=True)

    if cumulative_accuracy:
        df = calculate_accuracy_by_row(df, convert_large)

    if collect_time:
        df['collect_time'] = df['collect_time'].apply(convert_to_numeric)
        max_value = get_max_numeric(df['collect_time'].copy())
        df['collect_time'] = df['collect_time'].replace('Timed_out', max_value)
        df['collect_time'] = df['collect_time'].fillna(0)
        df['collect_time'] = pd.to_numeric(df['collect_time'])

    if remove_trivial:
        first_non_zero_index = df['Cum_Sum'].ne(0).idxmax()
        df = df.loc[first_non_zero_index:]

    df = df.reset_index(drop=True)
    baseline_time = df['Time'].iloc[0]
    df['Time_passed'] = df['Time'] - baseline_time
    return df


def calculate_accuracy_by_row(df:pd.DataFrame, convert_large=True):
    """calculate accuracy at each time stamp

    Args:
        df (pd.DataFrame): data from cleaned csv
        convert_large (bool): whether convert accuracy from 0-1 scale to 0-100 scale
    """    
    active_poke = df['Active_Poke'].loc[0]
    df['Percent_Correct'] = df[f'{active_poke}_Poke_Count'] / (df['Left_Poke_Count']+df['Right_Poke_Count'])
    
    if convert_large:
        df['Percent_Correct'] *= 100
        
    return df
        
    
def get_max_numeric(series:pd.Series):
    numeric_values = pd.to_numeric(series, errors='coerce')  # Convert non-numeric to NaN
    return numeric_values.max(skipna=True)

def get_retrieval_time(path, sheet, day=3):
    data = read_excel_by_sheet(sheet, path)
    data = data[data['Time_passed'] < timedelta(days=day)]

    times = data['collect_time'].tolist()
    pellet_times = [each for each in times if each != 0.0]
    pellet_times = list(map(float, pellet_times))
    pellet_times = [each for each in pellet_times if not math.isnan(each)]
    return pellet_times

def find_dispense_time(df):
    total_dispense = 0 # in minutes
    start_dispense = df['Time'][0] if 'Dispense' in df['Event'][0] else None

    for idx, row in df.iterrows():
        if idx == len(df) - 1: continue
        # dispense in next row and not in current row -> dispense start
        if 'Dispense' not in row['Event'] and 'Dispense' in df.loc[idx+1, 'Event']: 
            start_dispense = row['Time']
        # dispense in current event but not the next -> dispense end
        elif 'Dispense' in row['Event'] and 'Dispense' not in df.loc[idx+1, 'Event']:
            total_dispense += (df.loc[idx+1, 'Time'] - start_dispense).total_seconds() / 60
    return round(total_dispense, 3)

def find_dispense_time_one_day(path, sheet):
    df = pd.read_excel(path, sheet).rename(columns={
            'MM:DD:YYYY hh:mm:ss': 'Time', 'Retrieval_Time': 'collect_time'})
    df['Time'] = pd.to_datetime(df['Time'])
    baseline_time = df['Time'].iloc[0]
    df['Time_passed'] = df['Time'] - baseline_time
      
    # preserve one day only
    target_time = timedelta(hours=24)
    df['time_diff'] = (df['Time_passed'] - target_time).abs()
    df = df[:df['time_diff'].idxmin()].reset_index(drop=True)
    
    return find_dispense_time(df)


def find_dispense_time_by_day(path, sheet):
    df = pd.read_excel(path, sheet).rename(columns={
            'MM:DD:YYYY hh:mm:ss': 'Time', 'Retrieval_Time': 'collect_time'})
    df['Time'] = pd.to_datetime(df['Time'])
    baseline_time = df['Time'].iloc[0]
    df['Time_passed'] = df['Time'] - baseline_time
    
    dispenses = []
      
    # preserve one day only
    df['time_diff_1'] = (df['Time_passed'] - timedelta(days=1)).abs()
    df['time_diff_2'] = (df['Time_passed'] - timedelta(days=2)).abs()
    df['time_diff_3'] = (df['Time_passed'] - timedelta(days=3)).abs()
    day_1 = df['time_diff_1'].idxmin()
    day_2 = df['time_diff_2'].idxmin()
    day_3 = df['time_diff_3'].idxmin()

    dispenses.append(find_dispense_time(df[:day_1].reset_index(drop=True)))
    if day_1 == day_2:
        dispenses.append(find_dispense_time(df[day_1:].reset_index(drop=True)))
        dispenses.append(-1)
        return dispenses
    else:
        dispenses.append(find_dispense_time(df[day_1:day_2].reset_index(drop=True)))
    
    if day_2 == day_3:
        dispenses.append(find_dispense_time(df[day_2:].reset_index(drop=True)))
    else:
        dispenses.append(find_dispense_time(df[day_2:day_3].reset_index(drop=True)))
    
    return dispenses