import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import sys

import ssm
sys.path.append('..')
import meals as ml


# def model_training(path: str, num_states: int, obs_dim: int, num_categories: int,
#                     max_iter: int, feat: list()) -> tuple:
#     """
#     The feat, feature list, is corresponding to the order:
#     [curr_active, prev_event, prev_active, meal, prev_reward]
#     """
#     if len(feat) < 5:
#         print('Missing boolean parameters for features')
#         return
    
#     # calculate input dimension
#     input_dim = 1
#     for f in feat:
#         if f == True:
#             input_dim += 1

#     X, y, original = extract_features(path, feat[0], feat[1], feat[2], feat[3], feat[4])

#     model = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
#                     observation_kwargs=dict(C=num_categories), transitions="inputdriven")
    
#     log3 = model.fit(y, inputs=X, method='em', num_iters=max_iter, tolerance=10**-4)
    
#     return log3, model, X, y, original

def in_meal(meals: list, time) -> bool:
    """
    Given a list of time periods that mouse is during a meal, evaluate whether
    it is in meal at this moment
    """
    for each in meals:
        if time >= each[0] and time <= each[1]:
            return True
    return False


def extract_features(path: str, prev_trace: int, meal: bool) -> tuple:
    """
    extract current active poke, previous choice, previous active poke, previous reward and biasas X
    use current event as output
    """
    if prev_trace <= 0 or prev_trace > 6:
        raise ValueError(f'prev_trace needs to be integer from 0 to 6, but got {prev_trace}')
    
    data = pd.read_csv(path)
    data = data[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke', 'Pellet_Count']].replace(
        {'LeftWithPellet': 'Left', 'LeftDuringDispense': 'Left',
         'RightWithPellet': 'Right', 'RightDuringDispense': 'Right'}).rename(
        {'MM:DD:YYYY hh:mm:ss': 'Time'}, axis='columns')
    data['Time'] = pd.to_datetime(data['Time'])
    data = data[data['Event'] != 'Pellet'].reset_index(drop=True)
    meals = ml.find_meals(data)

    # Extract the date and time and take 2nd day
    data['Date'] = data['Time'].dt.date
    data['Time_of_day'] = data['Time'].dt.time
    unique_dates = data['Date'].unique()
    second_day = unique_dates[1]

    # Filter rows for the second day between 7:00 a.m. and 2:00 p.m.
    start_time = pd.to_datetime('00:00:00').time()
    end_time = pd.to_datetime('23:59:59').time()
    data = data[(data['Date'] == second_day) & (
        data['Time_of_day'] >= start_time) & (data['Time_of_day'] <= end_time)]

    
    # extract features
    mapper = {'Left': 1, 'Right': 0}
    bool_mapper = {True: 1, False: 0}
    data['Event'] = data['Event'].map(mapper)
    data['Active_Poke'] = data['Active_Poke'].map(mapper)

    for i in range(1, prev_trace+1):
        if i == 1:
            data['prev_1_event'] = data['Event'].shift(fill_value=None)
            data['prev_active'] = data['Active_Poke'].shift(fill_value=None)
            data['prev_1_reward'] = [row['prev_1_event'] == row['prev_active'] for idx, row in data.iterrows()]
        else:
            data[f'prev_{i}_event'] = data[f'prev_{i-1}_event'].shift(fill_value=None)
            data['prev_active'] = data['prev_active'].shift(fill_value=None)
            data[f'prev_{i}_reward'] = [row[f'prev_{i-1}_event'] == row['prev_active'] for idx, row in data.iterrows()]
        
        data[f'prev_{i}_reward'] = data[f'prev_{i}_reward'].map(bool_mapper)
        
    if meal:
        data['meal'] = [in_meal(meals, each) for each in data['Time']]
        data['meal'] = data['meal'].map(bool_mapper)

    data = data.drop(
        ['Date', 'Time_of_day', 'Time', 'prev_active', 'Active_Poke', 'Pellet_Count'], axis=1)

    X = data.drop(['Event'], axis='columns')[prev_trace:]
    y = data['Event'][prev_trace:]

    X['bias'] = 1
    feats = X.columns
    return X.to_numpy(), np.array(y), feats


# def graph_model_parameters(model: ssm.HMM, log_array: list, feat: list):
#     """
#     Graph GLM weights, transition matrix and log probability for the model
#     """
#     para = model.observations.params
#     tran = model.transitions.transition_matrix
#     num_states = model.K
#     input_dim = model.M

#     factor_dict = []
#     if feat[0]:
#         factor_dict = ['Curr Active']
#     if feat[1]:
#         factor_dict.append('Prev Event')
#     if feat[2]:
#         factor_dict.append('Prev Active')
#     if feat[3]:
#         factor_dict.append("Meal")
#     if feat[4]:
#         factor_dict.append('Prev Reward')

#     factor_dict.append('bias')

#     states = []
#     for i in range(1, num_states+1):
#         states.append(str(i))

#     # Plot parameters:
#     fig = plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
#     plt.subplot(1, 3, 1)
#     cols = ['#377eb8', '#ff7f00', '#4daf4a', '#fc0303']
#     for k in range(num_states):
#         plt.plot(range(input_dim), para[k][0], marker='o',
#                  color=cols[k], linestyle='-',
#                  lw=1.5, label="state " + str(k+1))
#     plt.yticks(fontsize=10)
#     plt.ylabel("GLM weight", fontsize=15)
#     plt.xlabel("covariate", fontsize=15)
#     plt.xticks(range(input_dim), factor_dict, fontsize=12, rotation=45)
#     plt.axhline(y=0, color="k", alpha=0.5, ls="--")
#     plt.legend()
#     plt.title("Model weights", fontsize=15)

#     plt.subplot(1, 3, 2)
#     gen_trans_mat = tran
#     plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
#     for i in range(gen_trans_mat.shape[0]):
#         for j in range(gen_trans_mat.shape[1]):
#             text = plt.text(j, i, str(np.around(gen_trans_mat[i, j], decimals=3)), ha="center", va="center",
#                             color="k", fontsize=12)
#     plt.xlim(-0.5, num_states - 0.5)
#     plt.xticks(range(0, num_states), states, fontsize=10)
#     plt.yticks(range(0, num_states), states, fontsize=10)
#     plt.ylim(num_states - 0.5, -0.5)
#     plt.ylabel("state t", fontsize=15)
#     plt.xlabel("state t+1", fontsize=15)
#     plt.title("Model transition matrix", fontsize=15)

#     plt.subplot(1, 3, 3)
#     plt.plot(log_array)
#     plt.title("Log Likelihood", fontsize=15)
#     plt.xlabel('Iteration', fontsize=15)
#     plt.show()


# def display_fitting_results(model: ssm.HMM, X, y):
#     """
#     Graph the occupancy percentage of each state
#     print out model accuracy when predicting actual behavior,
#     model accuracy in each state (compare model prediction of event and active poke) and overall accuracy
#     """
#     pred_state, pred_choice = model.sample(len(X), input=X)
#     num_state = model.K

#     state_list = []
#     for i in range(num_state):
#         state_list.append(f'State {i+1}')

#     # Model Prediction vs Actual Sequence
#     accuracy = accuracy_score(y, pred_choice)

#     # Model state occupancy
#     corr_state_acc = [0] * num_state
#     overall_acc = 0
#     states = [0] * num_state
#     for state in pred_state:
#         states[state] += 1
#     states_percent = [i/len(X) for i in states]

#     # Accuracy w.r.t states
#     for idx, row in enumerate(X):
#         # row[0] is active poke
#         if row[0] == pred_choice[idx]:
#             corr_state_acc[pred_state[idx]] += 1
#             overall_acc += 1

#     for i in range(num_state):
#         corr_state_acc[i] /= states[i]

#     # Plot state occupancy
#     plt.figure(figsize=(6, 6))
#     sns.barplot(x=state_list,
#                 y=states_percent, width=0.6, palette='bright')
#     plt.title(f'State Occupancy in {num_state} State Model', fontsize=18)
#     plt.ylabel('Occupancy Rate', fontsize=16)
#     plt.show()

#     # Overall accuracy in the model
#     print('Accuracy of Predicting Mice Behavior:', accuracy)
#     for i in range(num_state):
#         print(f'State {i+1} Percentage:',
#           states_percent[i], '; Accuracy:', corr_state_acc[i])
#     print('Overall Accuracy in Model:', overall_acc/len(X))
