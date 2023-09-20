import pandas as pd
import os


def combine_csv(parent: str) -> pd.DataFrame:
    # old version cleaning
    output = os.path.join(parent, 'combined.csv')
    if os.path.exists(output):
        os.remove(output)

    # process file list
    files = os.listdir(path=parent)
    files.remove('.DS_Store')
    for i in range(len(files)):
        files[i] = os.path.join(parent, files[i])

    # concat csv files
    result = pd.DataFrame()
    length = []
    for file in files:
        curr = pd.read_csv(file)    # read current file

        curr = curr[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke']].rename(
            {'MM:DD:YYYY hh:mm:ss': 'Time_Stamp'})

        curr = curr.replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                             'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})

        length.append(curr.shape[0])
        if result.empty:
            result = curr
        else:
            result = pd.concat([result, curr], ignore_index=True)

    print("Files have length", length)
    print("Total length should be", sum(length))

    # return and save results
    result.to_csv(output)
    return result


def count_error_rate(data: pd.DataFrame) -> tuple:
    error_R = 0
    error_L = 0
    total_R = 0
    total_L = 0

    for idx, row in data.iterrows():
        active_poke = row['Active_Poke']
        event = row['Event']

        if active_poke == 'Left':   # left active
            if event != 'Pellet':   # pellet and correct are overlapping
                total_L += 1

            if event == 'Right':
                error_L += 1
        else:       # right active
            if event != 'Pellet':
                total_R += 1

            if event == 'Left':
                error_R += 1

    rateL = round(error_L / total_L, 2)
    rateR = round(error_R / total_R, 2)

    return rateL, rateR


def preprocess_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data[['Event', 'Active_Poke']]

    data = data.replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                         'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})

    return data


def parent_directory_process(parent: str):
    files = os.listdir(path=parent)
    files.remove('.DS_Store')

    for i in range(len(files)):
        files[i] = os.path.join(parent, files[i])

    return files


# path = '../behavior data integrated/Bhv 5 - Ctrl/M1/Contingency Flip/FED000_071123_00.CSV'
def get_bhv_num(path: str) -> tuple:
    branches = path.split(sep='/')
    num = branches[3][1]
    bhv = branches[2][4]

    return bhv, num
