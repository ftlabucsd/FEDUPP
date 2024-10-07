from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import meals as ml
from preprocessing import read_csv_clean
import os
import pickle

def label_indices(labels):
    result = defaultdict(list)
    for idx, val in enumerate(labels):
        result[val].append(idx)
    return result


def index2meal(data_div: defaultdict, data:list):
    data = np.array(data)
    meal_by_category = defaultdict()
    for key, val in data_div.items():
        meal_by_category[key] = data[val]
    return meal_by_category


def extract_data_full_group(file_paths:list):
    data = defaultdict(list)
    for path in file_paths:
        each = read_csv_clean(path, remove_trivial=False, collect_time=True)
        each_acc_dict = ml.extract_meals_data(data=each, 
                                            time_threshold=60,
                                            pellet_threshold=2)

        for key, item in each_acc_dict.items():
            data[key].extend(item)

    ml.print_meal_stats(data)
    # ctrl_data = ml.preprocess_meal_data(ctrl_data)
    return data


def find_k_by_elbow(data:list):
    k_values = range(1, 8)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Plotting the elbow curve
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, inertias, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()
    
    
def fit_model_single(data:list, k:int):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_

    data_div = label_indices(labels)
    meals_by_category = index2meal(data_div, data)
    score = silhouette_score(data, kmeans.labels_)
    print("Silhouette Score:", score)
    visualize_kmeans(data, labels)
    
    return kmeans, meals_by_category

def visualize_kmeans(data:list, labels:list):
    # Reducing dimensionality for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Plotting
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('Cluster visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster Label')
    plt.show()


def read_data(filename:str) -> list:
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
    else:
        data = []

    return data


def update_data(filename:str, new_list:np.array):
    data = read_data(filename)
    print(f'Old data has {len(data)} items')
    data.extend(new_list.tolist())
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f'New data has {len(data)} items')


def collect_meals_from_categories(meals_by_category:dict, good_class:list):
    bad_class = [each for each in meals_by_category.keys() if each not in good_class]
    good_meals = []
    bad_meals = []

    for idx in good_class:
        good_meals.extend(meals_by_category[idx])
    for idx in bad_class:
        bad_meals.extend(meals_by_category[idx])
    return np.array(good_meals), np.array(bad_meals)


def data_padding(data: list) -> np.array:
    for each in data:
        size = len(each)
        while size < 4:
            each.append(-1)
            size += 1
    return np.array(data)

def create_dataset_single_group(experiment:str, ctrl:bool):
    data_root = f'{experiment}_{"ctrl" if ctrl else "exp"}_'

    good_X = read_data(data_root+'good.pkl')
    bad_X = read_data(data_root+'bad.pkl')
    good_y = np.zeros((len(good_X)))
    bad_y = np.ones((len(bad_X)))

    X = np.vstack((data_padding(good_X), data_padding(bad_X)))
    y = np.concatenate((good_y, bad_y))

    return X, y


def merge_dataset(ctrl_X:np.array, ctrl_y:np.array, exp_X:np.array, exp_y:np.array):
    X = np.vstack((ctrl_X, exp_X))
    y = np.concatenate(((ctrl_y, exp_y)))
    return X, y
