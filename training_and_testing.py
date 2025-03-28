# %%
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
from nelson_langer_network import build_nelson_network
import time
from astrochem_clustering import ClusterTree, Cluster, AstrochemClusterModel

# Load parameters
datafile = np.load('data/tracer_parameter_data.npy')
datafile[:,0] *= 2
# ['nH', 'T', 'XH2', 'FUV', 'NH', 'zeta'] are the columns of datafile
data = datafile[:,[0, 1, 3]]
# data = datafile[:,[0, 1]]
data[:,[0,1]] = np.log10(data[:,[0,1]])
# data[:,2] /= 8.94e-14
params = pd.DataFrame(data=data, columns=['$\log(n_h)$', '$\log(T)$', '$G_0$'])

# Normalize params dataframe
mean = params.mean()
std = params.std()
params_normalized = (params - mean) / std

# Data for ode solves
x0 = np.array([
    0.5,
    9.059e-9,
    2e-4,
    0.1,
    7.866e-7,
    0.0,
    0.0,
    0.0004,
    0.0,
    0.0,
    0.0,
    0.0002,
    2.0e-7,
    2.0e-7
])
secs_per_year = 3600*24*365
tf = 100000 * secs_per_year


#%% Build surrogate model

def compute_clusters(
    params: pd.DataFrame,
    error_tol: float,
    QoI: int,
    time: float,
    N: int = 10, # number of initial clusters to split into
    ss: int = 40, # sample size used to compute statistics in each cluster
    ) -> int:
    # Computes the clusters recursively, and makes smaller based on error_tol.

    k = 0
    indices = np.full(shape=params.shape[0], fill_value=-1)

    kmeans = cluster.KMeans(n_clusters=N, random_state=1)
    labels = kmeans.fit_predict(params[['$\log(n_h)$', '$\log(T)$', '$G_0$']])
    centroids = kmeans.cluster_centers_
    cluster_structure = np.empty(shape=N, dtype=object)

    for j in range(N):

        # Compute error inside each cluster
        params_in_cluster = params[labels==j]

        ss_new = ss
        if params_in_cluster.shape[0] < ss:
            # Cluster error is not converging in this case. This is a big issue
            ss_new = params_in_cluster.shape[0]
        
        param_sample = params_in_cluster.sample(ss_new)
        centroid_params = centroids[j]
        qc = solve_nelson_network(centroid_params, x0, QoI, time)

        # Compute max statistic via max_{j=1,...,ss}(q_c - q_j) / q_c
        dvec = np.empty(shape=ss_new)
        for i, row in param_sample.reset_index(drop=True).iterrows():
            qi = solve_nelson_network(row.to_numpy(), x0, QoI, time)
            dvec[i] = np.abs(qc-qi)
        error = np.max(dvec) / qc

        if error > error_tol:
            # Split cluster in half (recursive step)
            k, cluster_structure[j] = compute_clusters_recursion_helper(
                    params_in_cluster, error_tol, QoI, time, indices, ss=ss, k=k)
        else:
            # Tolerance is good; save values in cluster column
            indices[params_in_cluster.index.to_numpy()] = k
            cluster_structure[j] = Cluster(k, centroid_params, qc)
            k += 1

    return k, indices, cluster_structure


def compute_clusters_recursion_helper(
    params: pd.DataFrame,
    error_tol: float,
    QoI: int,
    time: float,
    indices: np.ndarray, # array containing cluster labels (reference)
    ss: int = 40, # sample size used to compute statistics in each cluster
    k: int = 0 # current cluster index (for recursive purposes)
    ) -> int:

    N = 2  # split in half
    kmeans = cluster.KMeans(n_clusters=N)
    labels = kmeans.fit_predict(params[['$\log(n_h)$', '$\log(T)$', '$G_0$']])
    centroids = kmeans.cluster_centers_

    # First cluster (N=0)
    params_in_cluster_0 = params[labels==0]

    ss_new = ss
    if params_in_cluster_0.shape[0] < ss:
        # Cluster error is not converging in this case. This is a big issue
        # Can alternatively just use this as a cluster
        ss_new = params_in_cluster_0.shape[0]

    param_sample = params_in_cluster_0.sample(ss_new)
    centroid_params = centroids[0]
    qc = solve_nelson_network(centroid_params, x0, QoI, time)

    # Compute max statistic via max_{j=1,...,ss}(q_c - q_j) / q_c
    dvec = np.empty(shape=ss_new)
    for i, row in param_sample.reset_index(drop=True).iterrows():
        qi = solve_nelson_network(row.to_numpy(), x0, QoI, time)
        dvec[i] = np.abs(qc-qi)
    error = np.max(dvec) / qc

    if error > error_tol:
        # Split cluster in half (recursive step)
        k, left = compute_clusters_recursion_helper(
                params_in_cluster_0, error_tol, QoI, time, indices, ss=ss, k=k)
    else:
        # Tolerance is good; save values in cluster column
        indices[params_in_cluster_0.index.to_numpy()] = k
        left = Cluster(k, centroid_params, qc)
        k += 1

    # Second cluster (N=1)
    params_in_cluster_1 = params[labels==1]

    ss_new = ss
    if params_in_cluster_1.shape[0] < ss:
        # Cluster error is not converging in this case. This is a big issue
        ss_new = params_in_cluster_1.shape[0]
    
    param_sample = params_in_cluster_1.sample(ss_new)
    centroid_params = centroids[1]
    qc = solve_nelson_network(centroid_params, x0, QoI, time)

    # Compute max statistic via max_{j=1,...,ss}(q_c - q_j) / q_c
    dvec = np.empty(shape=ss_new)
    for i, row in param_sample.reset_index(drop=True).iterrows():
        qi = solve_nelson_network(row.to_numpy(), x0, QoI, time)
        dvec[i] = np.abs(qc-qi)
    error = np.max(dvec) / qc

    if error > error_tol:
        # Split cluster in half (recursive step)
        k, right = compute_clusters_recursion_helper(
                params_in_cluster_1, error_tol, QoI, time, indices, ss=ss, k=k)
    else:
        # Tolerance is good; save values in cluster column
        indices[params_in_cluster_1.index.to_numpy()] = k
        right = Cluster(k, centroid_params, qc)
        k += 1

    return k, ClusterTree(None, left, right)



def solve_nelson_network(params_row: np.ndarray, x0: np.ndarray, QoI: int, time: float):
    n_h = 10**params_row[0]
    T = 10**params_row[1]
    G0 = params_row[2]
    network = build_nelson_network(n_h = n_h, T = T, G0 = G0)
    tvec, yvec = network.solve_reaction([0, time], x0)
    return yvec[QoI, -1]

def flatten_cluster_centers(tree, N_clusters):
    centroids = np.zeros(shape=(N_clusters, 3))
    QoI_values = np.zeros(shape=N_clusters)
    k = 0
    for val in tree:
        if type(val) == Cluster:
            centroids[k] = val.params
            QoI_values[k] = val.QoI_value
            k += 1
        elif type(val) == ClusterTree:
            k = flatten_cluster_centers_recursion(val.left, k, centroids, QoI_values)
            k = flatten_cluster_centers_recursion(val.right, k, centroids, QoI_values)
    return centroids, QoI_values

def flatten_cluster_centers_recursion(tree, k, centroids, QoI_values):
    # left
    if type(tree) == Cluster:
        centroids[k] = tree.params
        QoI_values[k] = tree.QoI_value
        k += 1
    elif type(tree) == ClusterTree:
        k = flatten_cluster_centers_recursion(tree.left, k, centroids, QoI_values)
        k = flatten_cluster_centers_recursion(tree.right, k, centroids, QoI_values)
    return k


# %% splitting testing data and training data, then train model
msk = np.random.rand(len(params)) > 0.2
train = params[msk]
test = params[~msk]

print('Starting training...')
start_time = time.perf_counter()
N_clusters, labels, tree = compute_clusters(train.reset_index(drop=True), 0.1, 9, tf, N=10)
print('Model finished training')
end_time = time.perf_counter()
centroids, QoI_values = flatten_cluster_centers(tree, N_clusters)

# Save model
model_data = np.zeros(shape=(N_clusters,4))
model_data[:,[0,1,2]] = centroids
model_data[:,3] = QoI_values
np.save('data/cluster_model.npy', model_data)


# %% try with object
msk = np.random.rand(len(params)) > 0.2
train = params[msk]
test = params[~msk]
start_time = time.perf_counter()
surrogate = AstrochemClusterModel(train.reset_index(drop=True))
surrogate.train_surrogate_model(0.1, 9, x0, tf, 10, 40)
end_time = time.perf_counter()
total_time = end_time - start_time # in seconds
print(f'Training time: {total_time:.2f} seconds')
centroids, QoI_values = surrogate.flatten_cluster_centers()


# %% If we want to save the surrogate, we need to use pickle 
import pickle
with open('data/high_res_model.pkl', 'wb') as file:
    pickle.dump(surrogate, file)

# %% Load the tree back in
import pickle
with open('data/surrogate.pkl', 'rb') as file:
    surrogate = pickle.load(file)

# %% Test training data
start_time = time.perf_counter()
datamat = np.empty(shape=(len(test),3))
# col1 is predicted value, col2 is actual value, col3 is percent error
k = 0
for index, row in test.reset_index(drop=True).iterrows(): # row is the parameters
    dvec = np.zeros(centroids.shape[0])
    j = 0
    # This is slow!!!!!! We can use the cluster tree structure (needs to be implemented)
    for centroid in centroids:
        dvec[j] = np.linalg.norm(centroid - row.to_numpy())
        j += 1
    predicted_cluster = np.argmin(dvec)
    datamat[k,0] = solve_nelson_network(row.to_numpy(), x0, 9, tf)
    datamat[k,1] = QoI_values[predicted_cluster]
    k += 1

end_time = time.perf_counter()
print(f'Time to run all testing data: {end_time-start_time} seconds')
datamat[:,2] = np.abs(datamat[:,1] - datamat[:,0]) / np.abs(datamat[:,1])


# Save testing data
testing_data_save = np.zeros(shape=(len(test),6))
testing_data_save[:,[0,1,2]] = test.reset_index(drop=True).to_numpy()
testing_data_save[:,[3,4,5]] = datamat
# cols 1-3 are params, col 4 is exact value (from ODE), col 5 is predicted, col 6 is percent error |col4 - col5|/col4

np.save('data/testdata_with_error.npy', datamat)


# %% try with object

start_time = time.perf_counter()
datamat = np.empty(shape=(len(test),3))
for index, row in test.reset_index(drop=True).iterrows():
    datamat[index, 0] = solve_nelson_network(row.to_numpy(), x0, 9, tf)
mid_time = time.perf_counter()
datamat[:,1] = surrogate.predict(test.to_numpy())
end_time = time.perf_counter()
print(f'Time to solve all ODEs: {mid_time-start_time} seconds')
print(f'Time to predict data: {end_time-mid_time}')
datamat[:,2] = np.abs(datamat[:,1] - datamat[:,0]) / np.abs(datamat[:,1])

# %% Error statistics
print(f'Mean: {np.mean(datamat[:,2])}')
print(f'Median: {np.median(datamat[:,2])}')
print(f'Max: {np.max(datamat[:,2])}')
print(f'STD: {np.std(datamat[:,2])}')
num_vals_outside_error = (datamat[:,2] > 0.1).sum()
print(f'# of points outside error: {num_vals_outside_error} ({num_vals_outside_error/len(datamat[:,2])}% of data)')

# %%
