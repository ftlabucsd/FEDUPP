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