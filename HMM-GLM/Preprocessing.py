import pandas as pd


def extract_features(path: str) -> tuple:
    data = pd.read_csv(path)
    data = data[['Event', 'Active_Poke']].replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                                        'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})

    data = data[data['event'] != 'pellet']
    data['prev_event'] = data['Event'].shift(fill_value=None)
    data['prev_active'] = data['Active_Poke'].shift(fill_value=None)
    # TODO: encode left and right with 0 and 1
    if not data.empty:
        data = data.iloc[1:]

    X = data.drop(['Event'], axis='columns')
    y = data['Event']
    return X.values.tolist(), y.values.tolist()
