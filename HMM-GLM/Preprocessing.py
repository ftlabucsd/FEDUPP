import pandas as pd
import numpy as np

def extract_features(path: str) -> tuple:
    """
    extract current active poke, previous choice, previous active poke, and biasas X
    use current event as output
    """
    data = pd.read_csv(path)
    data = data[['Event', 'Active_Poke']].replace({'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
                                        'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'})

    data = data[data['Event'] != 'Pellet']
    data['prev_event'] = data['Event'].shift(fill_value=None)
    data['prev_active'] = data['Active_Poke'].shift(fill_value=None)
    if not data.empty:
        data = data.iloc[1:]

    X = data.drop(['Event'], axis='columns')
    y = data['Event']

    mapper = {'Left': 0, 'Right': 1, 'Pellet': 2}
    X['Active_Poke'] = X['Active_Poke'].map(mapper)
    X['prev_event'] = X['prev_event'].map(mapper)
    X['prev_active'] = X['prev_active'].map(mapper)
    X['bias'] = 1
    y = y.map(mapper)
    Y = []
    for each in y:
        Y.append([each])
    return X.to_numpy(), np.array(Y)
