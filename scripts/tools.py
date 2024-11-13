import pandas as pd
import os


def parent_directory_process(parent: str):
    files = os.listdir(path=parent)
    files.remove('.DS_Store')

    for i in range(len(files)):
        files[i] = os.path.join(parent, files[i])

    return files


def get_bhv_num(sheet: str) -> tuple:
    parts = sheet.split('.')
    return [parts[0][1:], sheet[-1]]


def get_session_time(data: pd.DataFrame) -> float:
    """Return session time of a mice data in hours

    Args:
        data (pd.DataFrame): behaviorial data

    Returns:
        float: duraion in hours
    """
    if 'Time_passed' in data.columns:
        return data['Time_passed'].iloc[-1].total_seconds() / 3600

    diff = data['Time'].loc[len(data) - 1] - data['Time'].loc[0]
    return diff.total_seconds() / 3600