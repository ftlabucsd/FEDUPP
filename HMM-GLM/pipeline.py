import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import KFold
import math

sys.path.append('../scripts')
import meals as ml
sys.path.append('./glmhmm')
import glm_hmm
from utils import find_best_fit, permute_states

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


def fit_all(model:glm_hmm.GLMHMM, X:np.array, y:np.array, inits=2, fit_init_states=False) -> tuple:
    """Fit GLMHMM model for all data

    Args:
        model (glm_hmm.GLMHMM): model to be trained
        inits (int, optional): Number of random initialization to try. Defaults to 2.
        fit_init_states (bool): Whether fit inital states. Defaults to False.
        
    Returns:
        tuple: log-likelihood, transition matrices, weights, initial state probs, trined model, best-performance index
    """
    # Initialization for all inits
    K, D, C = model.k, model.d, model.c
    lls_all = np.zeros((inits,500))
    A_all = np.zeros((inits,K,K))
    w_all = np.zeros((inits,K,D,C))

    for i in range(inits):
        A_init,w_init,pi_init = model.generate_params() # initialize the model parameters
        lls_all[i,:],A_all[i,:,:],w_all[i,:,:],pi0 = model.fit(y,X,A_init,w_init, 
                                                               tol=1e-5, maxiter=500,
                                                               fit_init_states=fit_init_states) # fit the model
        print('initialization %s complete' %(i+1))
    return lls_all, A_all, w_all, pi0, model, find_best_fit(lls_all)


def evaluate_likelihood(model:glm_hmm.GLMHMM, X:np.array, y:np.array, matrix:np.array, weight: np.array):
    phi = np.array([model.glm.compObs(X, weight[k]) for k in range(model.k)])
    phi = phi.transpose(1, 0, 2)
    fit_ll,_,_,_ = model.forwardPass(y,matrix,phi)
    return fit_ll


def graph_fit_all(A_all:np.array, w_all:np.array, lls_all:np.array, features:list, best_ll:float, bestix:int):
    num_states = A_all[0].shape[0]
    n_inits = lls_all.shape[0]
    label = [f"state {state+1}" for state in range(num_states)]
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#fc0303']
    
    # for easy comparison permute the states in order from highest to lowest self-transition probability
    A_permuted, order = permute_states(A_all[bestix])
    w_permuted,_ = permute_states(w_all[bestix],method='order',param='weights',order=order)
    
    plt.figure(figsize=(18, 6))
    # transition matrix
    plt.subplot(1, 3, 1)
    plt.imshow(A_permuted, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(num_states):
        for j in range(num_states):
            text = plt.text(j, i, str(np.around(A_permuted[i, j], decimals=3)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), label, fontsize=10)
    plt.yticks(range(0, num_states), label, fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.ylabel("state t", fontsize=15)
    plt.xlabel("state t+1", fontsize=15)
    plt.title("Model transition matrix", fontsize=15)
    
    # model weight
    plt.subplot(1, 3, 2)
    for i in range(num_states):
        plt.plot(w_permuted[i,:, 1], linestyle='-', color=colors[i], label=label[i], linewidth=2)
        
    plt.ylabel('weight', fontsize=15)
    plt.plot(range(len(features)), np.zeros(len(features)), linestyle='--', color='gray')  # Ensure alignment by using range(len(features))
    plt.xticks(range(len(features)), features, rotation=75)
    plt.xlabel("Features", fontsize=15)
    plt.title('Model Weight By Feature', fontsize=15)
    plt.legend()
    
    
    plt.subplot(1,3,3)
    best_line = lls_all[bestix]
    conv_idx = len(best_line) - 1
    
    for i, each in enumerate(best_line):
        if math.isnan(each): 
            conv_idx = i - 3
            break
        
    plt.plot(best_line, linewidth=2.5, color='blue', label='Best fit') # plot best fit
    
    for i in range(n_inits):
        if i != bestix:
            plt.plot(lls_all[i], color='gray', label='Other fits')

    plt.annotate(f'Best Likelihood: {best_ll:.2f}', 
            xy=(conv_idx, best_ll), 
            xytext=(-90, -90),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', lw=1.5),
            fontsize=12,
            color='blue')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Log Likelihood", fontsize=15)
    plt.xlabel('Iteration', fontsize=15)
    
    plt.tight_layout()
    plt.show()


def fit_cv(model:glm_hmm.GLMHMM, X:np.array, y:np.array, folds:int, inits:int):
    # split the data into five folds
    y_train = [None] * folds
    y_test = [None] * folds
    x_train = [None] * folds
    x_test = [None] * folds
    kf = KFold(n_splits=folds)
    kf.get_n_splits(y)
    for i, (train_index, test_index) in enumerate(kf.split(y)):
        y_train[i], y_test[i] = y[train_index], y[test_index]
        x_train[i], x_test[i] = X[train_index], X[test_index]
    
    lls_all = []  # List to store log-likelihoods
    A_all = []    # List to store A matrices
    w_all = []    # List to store w matrices

    # fit the model for each training set and each initialization
    for i in range(folds):
        fold_lls = []
        fold_As = []
        fold_ws = []
        for j in range(inits):
            model.n = len(y_train[i]) # reset the number of data points in accordance with the size of the training set
            A_init, w_init, pi_init = model.generate_params()  # initialize the model parameters
            lls, A, w, pi0 = model.fit(y_train[i], x_train[i], A_init, w_init, fit_init_states=True)  # fit the model
            
            # Append results for this initialization
            fold_lls.append(lls)
            fold_As.append(A)
            fold_ws.append(w)
            print(f'Initialization {j+1} complete')

        # Append results for this fold
        lls_all.append(fold_lls)
        A_all.append(fold_As)
        w_all.append(fold_ws)
        print('fold %s complete' %(i+1))
    return model, lls_all, A_all, w_all, x_test, y_test


def cv_evaluate(model:glm_hmm.GLMHMM, x_test:list, y_test:list,
                 A_all:np.array, w_all:np.array, lls_all:np.array):
    fit_ll = []

    for i in range(len(y_test)):  # Assuming folds is the actual number of folds used, not hard-coded to 5
        model.n = len(y_test[i])  # Adjusting to the actual test set size for the current fold

        lls = np.array(lls_all[i])  # Convert list of log-likelihoods for current fold to numpy array for processing
        bestix = np.argmax(lls.mean(axis=1))  # Assuming mean log-likelihood is the criterion for best fit

        # Convert inferred weights into observation probabilities for each state
        phi = np.array([model.glm.compObs(x_test[i], w_all[i][bestix][k]) for k in range(K)])
        phi = phi.transpose(1, 0, 2)

        fit_log_likelihood,_,_,_ = model.forwardPass(y_test[i], A_all[i][bestix], phi)
        fit_ll.append(fit_log_likelihood)

    print('Inferred LL: %f' % np.mean(fit_ll))
    return fit_ll


def display_fitting_results(model: glm_hmm.GLMHMM, X, y):
    """
    Graph the occupancy percentage of each state
    print out model accuracy when predicting actual behavior,
    model accuracy in each state (compare model prediction of event and active poke) and overall accuracy
    """
    _, pred_choice, pred_state = model.generate_data_from_fit(model.w, model.A, X)
    pred_choice, pred_state = list(map(int, pred_choice)), list(map(int, pred_state))
    # print(pred_choice[:5], pred_state[:5])
    num_state = model.k

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
    plt.figure(figsize=(6, 6))
    sns.barplot(x=state_list,
                y=states_percent, width=0.6, palette='bright')
    plt.title(f'State Occupancy in {num_state} State Model', fontsize=18)
    plt.ylabel('Occupancy Rate', fontsize=16)
    plt.show()

    # Overall accuracy in the model
    print('Accuracy of Predicting Mice Behavior:', accuracy)
    for i in range(num_state):
        print(f'State {i+1} Percentage:',
          states_percent[i], '; Accuracy:', corr_state_acc[i])
    print('Overall Accuracy in Model:', overall_acc/len(X))
