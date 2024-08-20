import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import sys

import ssm
sys.path.append('..')
from pipeline import extract_features

cols = ['#377eb8', '#ff7f00', '#4daf4a', '#fc0303']


def model_training(path: str, num_states: int, obs_dim: int, num_categories: int,
                    max_iter: int, prev_trace=5, meal=True, transition='recurrent') -> tuple:
    """
    The feat, feature list, is corresponding to the order:
    [curr_active, prev_event, prev_active, meal, prev_reward]
    """
    if transition not in ['recurrent', 'inputdriven']:
        raise ValueError('Unsupport transition type')
    
    X, y, features = extract_features(path, prev_trace=prev_trace, meal=meal)
    y = y.reshape(-1, 1)

    # calculate input dimension
    input_dim = X.shape[1]

    model = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", init_method='kmeans',
                    observation_kwargs=dict(C=num_categories), transitions=transition)
    
    log3 = model.fit(y, inputs=X, method='em', num_iters=max_iter, tolerance=1e-4)
    
    return log3, model, X, y, features


def graph_model_parameters(model: ssm.HMM, log_array: list, feat: list):
    """
    Graph GLM weights, transition matrix and log probability for the model
    """
    para = model.observations.params
    tran = model.transitions.transition_matrix
    num_states = model.K
    input_dim = model.M

    states = []
    for i in range(1, num_states+1):
        states.append(str(i))

    # Plot parameters:
    fig = plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(1, 3, 1)
    for k in range(num_states):
        plt.plot(range(input_dim), para[k][0], marker='o',
                 color=cols[k], linestyle='-',
                 lw=1.5, label="state " + str(k+1))
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks(range(input_dim), feat, fontsize=12, rotation=70)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.legend()
    plt.title("Model weights", fontsize=15)

    plt.subplot(1, 3, 2)
    gen_trans_mat = tran
    plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(gen_trans_mat.shape[0]):
        for j in range(gen_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(gen_trans_mat[i, j], decimals=3)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), states, fontsize=10)
    plt.yticks(range(0, num_states), states, fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.ylabel("state t", fontsize=15)
    plt.xlabel("state t+1", fontsize=15)
    plt.title("Model transition matrix", fontsize=15)

    plt.subplot(1, 3, 3)
    plt.plot(log_array)
    plt.title("Log Likelihood", fontsize=15)
    plt.xlabel('Iteration', fontsize=15)
    plt.show()


def display_fitting_results(model: ssm.HMM, X, y):
    """
    Graph the occupancy percentage of each state
    print out model accuracy when predicting actual behavior,
    model accuracy in each state (compare model prediction of event and active poke) and overall accuracy
    """
    pred_state, pred_choice = model.sample(len(X), input=X)
    num_state = model.K

    state_list = []
    for i in range(num_state):
        state_list.append(f'State {i+1}')

    # Model Prediction vs Actual Sequence
    accuracy = accuracy_score(y, pred_choice)

    # Model state occupancy
    corr_state_acc = [0] * num_state
    overall_acc = 0
    states = [0] * num_state
    for state in pred_state:
        states[state] += 1
    states_percent = [i/len(X) for i in states]

    # Accuracy w.r.t states
    for idx, row in enumerate(X):
        # row[0] is active poke
        if row[0] == pred_choice[idx]:
            corr_state_acc[pred_state[idx]] += 1
            overall_acc += 1

    for i in range(num_state):
        corr_state_acc[i] /= states[i]

    # Plot state occupancy
    plt.figure(figsize=(4, 4))
    sns.barplot(x=state_list,
                y=states_percent, width=0.6, palette='bright')
    plt.title(f'State Occupancy in {num_state} State Model', fontsize=18)
    plt.ylabel('Occupancy Rate', fontsize=16)
    plt.show()

    # Overall accuracy in the model
    print(f'Accuracy of Prediction: {accuracy}; Accuracy in the model: {overall_acc/len(X)}')
    for i in range(num_state):
        print(f'State {i+1} Percentage: {states_percent[i]}, ; Accuracy:, {corr_state_acc[i]}')


def graph_posterior_state_prob(model:ssm.HMM, X, y):
    posterior_probs = model.expected_states(data=y, input=X)[0]

    plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
    for k in range(model.K):
        plt.plot(posterior_probs[:, k], label="State " + str(k + 1), lw=2,
                color=cols[k])
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize = 10)
    plt.xlabel("Num Events", fontsize = 15)
    plt.ylabel("p(state)", fontsize = 15)
    plt.title("Posterior Probability of State", fontsize=18)
    plt.legend()
    