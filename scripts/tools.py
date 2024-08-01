import pandas as pd
import os


def parent_directory_process(parent: str):
    files = os.listdir(path=parent)
    files.remove('.DS_Store')

    for i in range(len(files)):
        files[i] = os.path.join(parent, files[i])

    return files


# /home/ftlab/Desktop/For_Andy/behavior data integrated/CASK/reversal/ctrl/B5M1.CSV
def get_bhv_num(path_or_sheet: str) -> tuple:
    if len(path_or_sheet) < 8:
        bhv = path_or_sheet[1]
        num = path_or_sheet[-1]
        return [bhv, num]

    elif 'CASK' in path_or_sheet:
        branches = path_or_sheet.split(sep='/')
        M = branches[-1].index('M')
        dot = branches[-1].index('.')
        num = branches[-1][M+1:dot]
        bhv = branches[-1][1:M]
        return [bhv, num]

    elif 'IVSA' in path_or_sheet:
        branches = path_or_sheet.split(sep='/')
        num = branches[-3][:2]
        return [num]
    
    elif 'mPFC' in path_or_sheet:
        branches = path_or_sheet.split(sep='/')
        dot = branches[-1].index('.')
        num = branches[-1][1:dot]
        return [num]



def get_session_time(data: pd.DataFrame) -> float:
    """Return session time of a mice data in hours

    Args:
        data (pd.DataFrame): behaviorial data

    Returns:
        float: duraion in hours
    """
    
    diff = data['Time'].loc[len(data)-1] - data['Time'].loc[0]
    
    return diff.total_seconds() / 3600