
import numpy as np
import sklearn.cluster
import pandas as pd
import faiss
from collections.abc import Callable


# bi-tree structures that we use in the recursive algorithm implemented below
class ClusterTree:

    def __init__(self, params, left, right):
        self.left = left
        self.right = right
        self.params = params
    

class Cluster:

    def __init__(self, index, params, QoI_value):
        self.index = index
        self.params = params
        self.QoI_value = QoI_value

    def __str__(self):
        return f'k={self.index}\tn_h={10**self.params[0]:.3f}\tT={10**self.params[1]:.3f}\tG0={self.params[2]:.6f}'



def recursive_cluster_algorithm(
    training_data: np.ndarray, # each row contains a set of parameters used to cluster
    eval_function: Callable, # callable function; takes numpy array to numpy array
    error_tol: float,
    N: int = 10, # number of initial clusters to split into
    ss: int = 40, # sample size used to compute statistics in each cluster
    ode_solves: np.ndarray = None, # if this is not none, uses these as solves. Otherwise runs eval_function every time
    ):

    # normalize the training data
    mean = training_data.mean()
    std = training_data.std()
    training_data = (training_data - mean) / std

    k = 0 # counting number of clusters as we go

    kmeans = sklearn.cluster.KMeans(n_clusters=N)
    labels = kmeans.fit_predict(training_data)
    centroids = kmeans.cluster_centers_
    cluster_structure = np.empty(shape=N, dtype=object)

    for j in range(N):

        # Compute error inside each cluster
        params_in_cluster = training_data[labels==j]

        ss_new = ss
        if len(params_in_cluster) < ss:
            ss_new = len(params_in_cluster)
        
        centroid_params = centroids[j]
        qc = eval_function(centroid_params * std + mean)

        # Compute error via max_{j=1,...,ss}(q_c - q_j) / q_c
        # param_sample = params_in_cluster.sample(ss_new)
        # dvec = np.empty(shape=ss_new)
        # for i, row in param_sample.reset_index(drop=True).iterrows():
        #     qi = self._solve_nelson_network(row.to_numpy(), x0, QoI, time)
        #     dvec[i] = np.abs(qc-qi)
        # error = np.max(dvec) / qc

        # Compute error by taking largest distances away and averaging errors
        distances_from_cluster = np.linalg.norm(centroid_params - params_in_cluster, axis=1)
        largest_row_indices = np.argpartition(distances_from_cluster, -ss_new)[-ss_new:]
        dvec = np.empty(shape=(ss_new, len(qc)))
        for i, row in enumerate(params_in_cluster[largest_row_indices]):
            if ode_solves == None:
                qi = eval_function(row * std + mean)
            else:
                qi = ode_solves[largest_row_indices[i]]
            dvec[i,:] = np.abs(qc-qi)
        error = np.max(dvec / qc)

        if error > error_tol:
            # Split cluster in half (recursive step)
            k, cluster_structure[j] = _recursive_clustering_helper(
                    params_in_cluster, centroid_params, error_tol, eval_function,
                    ss, ode_solves[labels==j], k, mean, std)
        else:
            # Tolerance is good; save values in cluster column
            cluster_structure[j] = Cluster(k, centroid_params, qc)
            k += 1


    return k, cluster_structure


def _recursive_clustering_helper(
    params: np.ndarray,
    prev_centroid: np.ndarray,
    error_tol: float,
    eval_function: Callable,
    ss: int = 40, # sample size used to compute statistics in each cluster
    ode_solves: np.ndarray = None,
    k: int = 0, # current cluster index (for recursive purposes)
    mean: float = 0, # mean of data (for recursion)
    std: float = 1 # std of data (for recursion)
):

    N = 2  # split in half
    kmeans = sklearn.cluster.KMeans(n_clusters=N)
    labels = kmeans.fit_predict(params)
    centroids = kmeans.cluster_centers_

    # First cluster (N=0)
    params_in_cluster_0 = params[labels==0]

    ss_new = ss
    if len(params_in_cluster_0) < ss:
        # Cluster error is not converging in this case. This is an issue maybe
        ss_new = len(params_in_cluster_0)

    centroid_params = centroids[0]
    qc = eval_function(centroid_params * std + mean)

    # Compute error by taking largest distances away and averaging errors
    # can make this block wayyyy faster using either faiss or matrices
    distances_from_cluster = np.linalg.norm(centroid_params - params_in_cluster_0, axis=1)
    largest_row_indices = np.argpartition(distances_from_cluster, -ss_new)[-ss_new:]
    dvec = np.empty(shape=(ss_new, len(qc)))
    for i, row in enumerate(params_in_cluster_0[largest_row_indices]):
        if ode_solves == None:
            qi = eval_function(row * std + mean)
        else:
            qi = ode_solves[largest_row_indices[i]]
        dvec[i,:] = np.abs(qc-qi)
    error = np.max(dvec / qc)

    if error > error_tol:
        # Split cluster in half (recursive step)
        k, left = _recursive_clustering_helper(
                params_in_cluster_0, centroid_params, error_tol, eval_function,
                ss, ode_solves[labels==0], k, mean, std)
    else:
        # Tolerance is good; save values in cluster column
        left = Cluster(k, centroid_params, qc)
        k += 1

    # Second cluster (N=1)
    params_in_cluster_1 = params[labels==1]

    ss_new = ss
    if len(params_in_cluster_1) < ss:
        # Cluster error is not converging in this case. This is a big issue
        ss_new = params_in_cluster_1.shape[0]
    
    centroid_params = centroids[1]
    qc = eval_function(centroid_params)

    # Compute error by taking largest distances away and averaging errors
    # can make this block wayyyy faster using either faiss or matrices
    distances_from_cluster = np.linalg.norm(centroid_params - params_in_cluster_1[['$\log(n_h)$','$\log(T)$','$G_0$']].to_numpy(), axis=1)
    largest_row_indices = np.argpartition(distances_from_cluster, -ss_new)[-ss_new:]
    dvec = np.empty(shape=(ss_new, len(qc)))
    for i, row in enumerate(params_in_cluster_1[largest_row_indices]):
        if ode_solves == None:
            qi = eval_function(row * std + mean)
        else:
            qi = ode_solves[largest_row_indices[i]]
        dvec[i,:] = np.abs(qc-qi)
    error = np.max(dvec / qc)

    if (error > error_tol).any():
        # Split cluster in half (recursive step)
        k, right = _recursive_clustering_helper(
                params_in_cluster_1, centroid_params, error_tol, eval_function,
                ss, ode_solves[labels==1], k, mean, std)
    else:
        # Tolerance is good; save values in cluster column
        right = Cluster(k, centroid_params, qc)
        k += 1

    return k, ClusterTree(prev_centroid, left, right)

