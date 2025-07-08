
from types import NoneType
import numpy as np
import sklearn.cluster
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
    mean = training_data.mean(axis=0)
    std = training_data.std(axis=0)
    training_data_normalized = (training_data - mean) / std

    k = 0 # counting number of clusters as we go

    kmeans = sklearn.cluster.KMeans(n_clusters=N)
    labels = kmeans.fit_predict(training_data_normalized)
    centroids_normalized = kmeans.cluster_centers_
    centroids = centroids_normalized * std + mean
    cluster_structure = np.empty(shape=N, dtype=object)

    for j in range(N):

        # Compute error inside each cluster
        params_in_cluster = training_data[labels==j]
        if type(ode_solves) == NoneType:
            ode_solves_in_cluster = None
        else:
            ode_solves_in_cluster = ode_solves[labels==j]

        ss_new = ss
        if len(params_in_cluster) < ss:
            ss_new = len(params_in_cluster)
        
        centroid_params = centroids[j]
        qc = eval_function(centroid_params)

        # this is a quick fix for the issue of #s of QoI
        if type(qc) != np.ndarray:
            qc = np.array([qc])
        

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
            if type(ode_solves_in_cluster) == NoneType:
                qi = eval_function(row)
            else:
                qi = ode_solves_in_cluster[largest_row_indices[i]]
            dvec[i,:] = np.abs(qc-qi)
        error = np.max(dvec / qc)

        if error > error_tol:
            # Split cluster in half (recursive step)
            k, cluster_structure[j] = _recursive_cluster_algorithm_helper(
                    params_in_cluster, centroid_params, error_tol,
                    eval_function, ss, ode_solves_in_cluster, k)
        else:
            # Tolerance is good; save values in cluster column
            cluster_structure[j] = Cluster(k, centroid_params, qc)
            k += 1


    return k, cluster_structure



def _recursive_cluster_algorithm_helper(
    params: np.ndarray,
    prev_centroid: np.ndarray,
    error_tol: float,
    eval_function: Callable,
    ss: int = 40, # sample size used to compute statistics in each cluster
    ode_solves: np.ndarray = None,
    k: int = 0, # current cluster index (for recursive purposes)
):

    # normalize for clustering
    mean = params.mean(axis=0)
    std = params.std(axis=0)
    # quick fix to the issue if std is 0 in one column
    if (std == 0).any():
        std += (std == 0) * 1
    params_normalized = (params - mean) / std

    N = 2  # split in half
    kmeans = sklearn.cluster.KMeans(n_clusters=N)
    labels = kmeans.fit_predict(params_normalized)
    centroids_normalized = kmeans.cluster_centers_
    centroids = centroids_normalized * std + mean

    # First cluster (N=0)
    params_in_cluster_0 = params[labels==0]
    if type(ode_solves) == NoneType:
        ode_solves_in_cluster_0 = None
    else:
        ode_solves_in_cluster_0 = ode_solves[labels==0]

    ss_new = ss
    if len(params_in_cluster_0) < ss:
        # Cluster error is not converging in this case. This is an issue maybe
        ss_new = len(params_in_cluster_0)

    centroid_params = centroids[0]
    qc = eval_function(centroid_params)

    # this is a quick fix for the issue of #s of QoI
    if type(qc) != np.ndarray:
        qc = np.array([qc])


    # Compute error by taking largest distances away and averaging errors
    # can make this block wayyyy faster using either faiss or matrices
    distances_from_cluster = np.linalg.norm(centroid_params - params_in_cluster_0, axis=1)
    largest_row_indices = np.argpartition(distances_from_cluster, -ss_new)[-ss_new:]
    dvec = np.empty(shape=(ss_new, len(qc)))
    for i, row in enumerate(params_in_cluster_0[largest_row_indices]):
        if type(ode_solves_in_cluster_0) == NoneType:
            qi = eval_function(row)
        else:
            qi = ode_solves_in_cluster_0[largest_row_indices[i]]
        dvec[i,:] = np.abs(qc-qi)
    error = np.max(dvec / qc)

    if error > error_tol:
        # Split cluster in half (recursive step)
        k, left = _recursive_cluster_algorithm_helper(
                params_in_cluster_0, centroid_params, error_tol,
                eval_function, ss, ode_solves_in_cluster_0, k)
    else:
        # Tolerance is good; save values in cluster column
        left = Cluster(k, centroid_params, qc)
        k += 1

    # Second cluster (N=1)
    params_in_cluster_1 = params[labels==1]
    if type(ode_solves) == NoneType:
        ode_solves_in_cluster_1 = None
    else:
        ode_solves_in_cluster_1 = ode_solves[labels==1]

    ss_new = ss
    if len(params_in_cluster_1) < ss:
        # Cluster error is not converging in this case. This is a big issue
        ss_new = len(params_in_cluster_1)
    
    centroid_params = centroids[1]
    qc = eval_function(centroid_params)

    # this is a quick fix for the issue of #s of QoI
    if type(qc) != np.ndarray:
        qc = np.array([qc])

    # Compute error by taking largest distances away and averaging errors
    # can make this block wayyyy faster using either faiss or matrices
    distances_from_cluster = np.linalg.norm(centroid_params - params_in_cluster_1, axis=1)
    largest_row_indices = np.argpartition(distances_from_cluster, -ss_new)[-ss_new:]
    dvec = np.empty(shape=(ss_new, len(qc)))
    for i, row in enumerate(params_in_cluster_1[largest_row_indices]):
        if type(ode_solves_in_cluster_1) == NoneType:
            qi = eval_function(row)
        else:
            qi = ode_solves_in_cluster_1[largest_row_indices[i]]
        dvec[i,:] = np.abs(qc-qi)
    error = np.max(dvec / qc)

    if (error > error_tol).any():
        # Split cluster in half (recursive step)
        k, right = _recursive_cluster_algorithm_helper(
                params_in_cluster_1, centroid_params, error_tol,
                eval_function, ss, ode_solves_in_cluster_1, k)
    else:
        # Tolerance is good; save values in cluster column
        right = Cluster(k, centroid_params, qc)
        k += 1

    return k, ClusterTree(prev_centroid, left, right)



def flatten_cluster_centers(nc, nq, tree):
    centroids = np.zeros(shape=(nc, 3))
    QoI_values = np.zeros(shape=(nc, nq))
    k = 0
    for val in tree:
        if type(val) == Cluster:
            centroids[k] = val.params
            QoI_values[k,:] = val.QoI_value
            k += 1
        elif type(val) == ClusterTree:
            k = _flatten_cluster_centers_helper(val.left, k, centroids, QoI_values)
            k = _flatten_cluster_centers_helper(val.right, k, centroids, QoI_values)
    return centroids, QoI_values



def _flatten_cluster_centers_helper(tree, k, centroids, QoI_values):
    if type(tree) == Cluster:
        centroids[k] = tree.params
        QoI_values[k,:] = tree.QoI_value
        k += 1
    elif type(tree) == ClusterTree:
        k = _flatten_cluster_centers_helper(tree.left, k, centroids, QoI_values)
        k = _flatten_cluster_centers_helper(tree.right, k, centroids, QoI_values)
    return k