import pandas as pd

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

    if hundredize:
        df['Percent_Correct'] *= 100
    if convert_time:
        df['Time'] = pd.to_datetime(df['Time'])
    if remove_trival:
        mask = (df['Percent_Correct'] != 0) & (df['Cum_Sum'] != 0)
        df = df[mask]

    df = df.reset_index().drop(['index'], axis='columns')
    return df


def read_csv_clean(path, remove_trivial=True):
    """
    Read csv file
    """
    df = pd.read_csv(path)

    df = df[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke', 'Pellet_Count']].rename(columns={
        'MM:DD:YYYY hh:mm:ss': 'Time'}).dropna()
    df['Cum_Sum'] = df['Pellet_Count'] / max(df['Pellet_Count'])
    
    df = df.replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                    'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})

    df['Time'] = pd.to_datetime(df['Time'])
    df = df.reset_index().drop(['index'], axis='columns')
    
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
    acc = []
    match = []
    
    for idx in range(len(df)):
        actual = df['Event'][:idx].to_list()
        target = df['Active_Poke'][:idx].to_list()
        
        for x,y in zip(actual, target):
            if x != 'Pellet':
                match.append(1 if x == y else 0)
        matched = sum(match)
        
        if matched == 0:
            acc.append(0)
        else:
            acc.append(round(matched / len(match), 2))
    
    if convert_large:
        acc = [100*val for val in acc]
        
    df['Percent_Correct'] = acc
    return df