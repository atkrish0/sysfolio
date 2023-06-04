import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# Step 1: Define your inputs
# - Asset returns
returns = pd.read_csv('returns_data.csv', index_col=0)  # Replace 'returns_data.csv' with your data file

# Step 2: Perform hierarchical clustering

# Calculate the pairwise distance matrix
distance_matrix = squareform(1 - returns.corr())  # Convert correlation to distance

# Apply hierarchical clustering
linkage_matrix = linkage(distance_matrix, 'single')  # You can use other linkage methods

# Step 3: Perform hierarchical risk parity

def get_recursive_clusters(linkage_matrix, cluster_labels, cluster):
    clusters = [cluster]
    if cluster < len(linkage_matrix):
        left_cluster = int(linkage_matrix[cluster, 0])
        right_cluster = int(linkage_matrix[cluster, 1])
        clusters += get_recursive_clusters(linkage_matrix, cluster_labels, left_cluster)
        clusters += get_recursive_clusters(linkage_matrix, cluster_labels, right_cluster)
    else:
        clusters += [cluster_labels[cluster - len(linkage_matrix)]]
    return clusters

def get_quasi_diag(linkage_matrix, cluster_labels):
    return get_recursive_clusters(linkage_matrix, cluster_labels, len(linkage_matrix) * 2 - 2)

def get_ivp(covariance_matrix):
    inversed_var = 1 / np.diag(covariance_matrix)
    return inversed_var / np.sum(inversed_var)

def get_cluster_var(covariance_matrix, cluster_indices):
    sub_cov_matrix = covariance_matrix[np.ix_(cluster_indices, cluster_indices)]
    weights = get_ivp(sub_cov_matrix)
    return np.dot(weights, np.dot(sub_cov_matrix, weights))

def get_hrp_weights(returns, linkage_matrix):
    cluster_labels = dendrogram(linkage_matrix, no_plot=True)['ivl']
    sorted_indices = returns.columns[list(map(int, cluster_labels))]
    
    covariance_matrix = returns[sorted_indices].cov().values
    weights = get_ivp(covariance_matrix)
    
    return pd.Series(weights, index=sorted_indices)

weights = get_hrp_weights(returns, linkage_matrix)

# Step 4: Print results

print("Asset Weights:")
for asset, weight in weights.items():
    print(f"{asset}: {weight}")
