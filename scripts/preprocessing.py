"""
This script provides functions for preprocessing and cleaning FED3 behavioral data.
It includes utilities for reading Excel files, extracting information from sheet names,
and preparing the data for further analysis by handling timestamps, calculating
accuracy, and cleaning up event data.
"""
import pandas as pd
import math
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


def get_bhv_num(sheet: str) -> tuple:
    """Extracts behavior and mouse numbers from a sheet name.

    Args:
        sheet (str): The name of the sheet (e.g., 'C1.M1' or 'R1M10').

    Returns:
        tuple: A tuple containing the behavior group and mouse number.
    """
    if '.' in sheet:
        parts = sheet.split('.')
        return [parts[0][1:], sheet[-1]]
    else:
        # R1M10 -> 1 and 10
        return sheet[1], sheet[3:]

def convert_to_numeric(value):
    """Converts a value to a numeric type if it is a numeric string.

    Args:
        value: The value to convert.

    Returns:
        The value converted to numeric, or the original value.
    """
    if isinstance(value, str) and value.isnumeric():
        return pd.to_numeric(value)
    else:
        return value

def get_all_sheet_names(excel_path:str):
    """Gets all sheet names from an Excel file.

    Args:
        excel_path (str): The path to the Excel file.

    Returns:
        list: A list of all sheet names in the Excel file.
    """
    xls = pd.ExcelFile(excel_path)
    return  xls.sheet_names


def read_excel_by_sheet(sheet, parent, convert_large=False, collect_time=True,
                        cumulative_accuracy=True, remove_trivial=False):
    """Reads a specific sheet from an Excel file and performs preprocessing.

    This function reads a sheet, renames columns, standardizes event names,
    converts data types, and can optionally calculate accuracy and clean the data.

    Args:
        sheet (str): The name of the sheet to read.
        parent (str): The path to the parent Excel file.
        convert_large (bool, optional): If True, scales accuracy to 0-100. Defaults to False.
        collect_time (bool, optional): If True, processes the collection time column. Defaults to True.
        cumulative_accuracy (bool, optional): If True, calculates row-wise accuracy. Defaults to True.
        remove_trivial (bool, optional): If True, removes initial rows with no pellets. Defaults to False.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
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
    """Calculates the accuracy of pokes at each timestamp.

    Args:
        df (pd.DataFrame): The input DataFrame with poke counts.
        convert_large (bool, optional): If True, scales accuracy to 0-100. Defaults to True.

    Returns:
        pd.DataFrame: The DataFrame with an added 'Percent_Correct' column.
    """
    active_poke = df['Active_Poke'].loc[0]
    df['Percent_Correct'] = df[f'{active_poke}_Poke_Count'] / (df['Left_Poke_Count']+df['Right_Poke_Count'])
    
    if convert_large:
        df['Percent_Correct'] *= 100
        
    return df
        
    
def get_max_numeric(series:pd.Series):
    """Finds the maximum numeric value in a pandas Series, ignoring non-numeric values.

    Args:
        series (pd.Series): The Series to search.

    Returns:
        The maximum numeric value in the Series.
    """
    numeric_values = pd.to_numeric(series, errors='coerce')  # Convert non-numeric to NaN
    return numeric_values.max(skipna=True)

def get_retrieval_time(path, sheet, day=3):
    """Extracts and cleans the pellet retrieval times from a given sheet.

    Args:
        path (str): The path to the Excel file.
        sheet (str): The name of the sheet to read.
        day (int, optional): The number of days of data to include. Defaults to 3.

    Returns:
        list: A list of cleaned pellet retrieval times in minutes.
    """
    data = read_excel_by_sheet(sheet, path)
    data = data[data['Time_passed'] < timedelta(days=day)]

    times = data['collect_time'].tolist()
    pellet_times = [each for each in times if each != 0.0]
    pellet_times = list(map(float, pellet_times))
    pellet_times = [each for each in pellet_times if not math.isnan(each)]
    return pellet_times