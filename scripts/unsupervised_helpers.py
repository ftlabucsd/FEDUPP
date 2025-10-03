"""
This script provides helper functions for unsupervised learning tasks on FED3
meal data. It includes utilities for data extraction, clustering, visualization,
and dataset preparation for machine learning models.
"""
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from scripts.meals import analyze_meals
import os
import pickle

def label_indices(labels):
    """Groups indices by their corresponding labels.

    Args:
        labels (list or np.ndarray): A list of labels.

    Returns:
        defaultdict: A dictionary where keys are labels and values are lists of indices.
    """
    result = defaultdict(list)
    for idx, val in enumerate(labels):
        result[val].append(idx)
    return result


def index2meal(data_div: defaultdict, data:list):
    """Converts a dictionary of label indices to a dictionary of meals.

    Args:
        data_div (defaultdict): A dictionary mapping labels to indices.
        data (list): A list of all meals.

    Returns:
        defaultdict: A dictionary where keys are labels and values are lists of meals.
    """
    data = np.array(data)
    meal_by_category = defaultdict()
    for key, val in data_div.items():
        meal_by_category[key] = data[val]
    return meal_by_category

def extract_meal_sequences(session_list, time_threshold=60, pellet_threshold=2, counts=(3, 4, 5)):
    sequences = {cnt: [] for cnt in counts}
    session_ratios = []
    for session in session_list:
        meals_with_acc, good_mask, _ = analyze_meals(
            session.raw.copy(),
            time_threshold=time_threshold,
            pellet_threshold=pellet_threshold,
            model_type='cnn',
        )
        total = len(good_mask)
        ratio = float(good_mask.sum()) / total if total else 0.0
        session_ratios.append(ratio)
        for _, padded in meals_with_acc:
            valid = [value for value in padded if value != -1]
            pellet_cnt = len(valid) + 1
            if pellet_cnt in sequences:
                sequences[pellet_cnt].append(valid)
    return sequences, session_ratios


def find_k_by_elbow(data:list):
    """Uses the elbow method to find the optimal number of clusters (k) for KMeans.

    Args:
        data (list): The input data for clustering.
    """
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
    
    
def fit_model_single(data:list, k:int, visualize:bool=False):
    """Fits a KMeans model to the data and visualizes the clusters.

    Args:
        data (list): The input data for clustering.
        k (int): The number of clusters.

    Returns:
        tuple: A tuple containing the fitted KMeans model and a dictionary of meals by category.
    """
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_

    data_div = label_indices(labels)
    meals_by_category = index2meal(data_div, data)
    score = silhouette_score(data, kmeans.labels_)
    print("Silhouette Score:", score)
    if visualize:
        visualize_kmeans(data, labels)
    return kmeans, meals_by_category

def visualize_kmeans(data:list, labels:list):
    """Visualizes the results of KMeans clustering using PCA for dimensionality reduction.

    Args:
        data (list): The input data that was clustered.
        labels (list): The cluster labels for each data point.
    """
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
    """Reads data from a pickle file.

    Args:
        filename (str): The name of the pickle file.

    Returns:
        list: The data loaded from the file, or an empty list if the file doesn't exist.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
    else:
        data = []

    return data


def update_data(filename:str, new_list:np.array):
    """Appends new data to an existing pickle file.

    Args:
        filename (str): The name of the pickle file to update.
        new_list (np.array): The new data to append.
    """
    data = read_data(filename)
    print(f'Old data has {len(data)} items')
    data.extend(new_list.tolist())
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f'New data has {len(data)} items')


def collect_meals_from_categories(meals_by_category:dict, good_class:list):
    """Separates meals into 'good' and 'bad' categories based on cluster labels.

    Args:
        meals_by_category (dict): A dictionary of meals grouped by cluster label.
        good_class (list): A list of labels corresponding to 'good' meal clusters.

    Returns:
        tuple: A tuple of two numpy arrays: (good_meals, bad_meals).
    """
    bad_class = [each for each in meals_by_category.keys() if each not in good_class]
    good_meals = []
    bad_meals = []

    for idx in good_class:
        good_meals.extend(meals_by_category[idx])
    for idx in bad_class:
        bad_meals.extend(meals_by_category[idx])
    return np.array(good_meals), np.array(bad_meals)


def data_padding(data: list) -> np.array:
    """Pads each meal in a list to a fixed length of 4 with -1.

    Args:
        data (list): A list of meals, where each meal is a list of accuracies.

    Returns:
        np.array: A numpy array of the padded meals.
    """
    for each in data:
        size = len(each)
        while size < 4:
            each.append(-1)
            size += 1
    return np.array(data)