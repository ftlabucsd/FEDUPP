from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import meals as ml
from preprocessing import read_csv_clean

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
    