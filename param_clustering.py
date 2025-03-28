# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from nelson_langer_network import build_nelson_network
from matplotlib.backends.backend_pdf import PdfPages

# Load parameters
datafile = np.load('np-states.npy')
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


# %% Kmeans to generate clusters for fixed N; at cluster centers, solve ODEs
N = 50
kmeans = cluster.KMeans(n_clusters=N, random_state=1)
labels = kmeans.fit_predict(params_normalized)
cluster_centers_normalized = pd.DataFrame(kmeans.cluster_centers_, columns=['$\log(n_h)$', '$\log(T)$', '$G_0$'])
cluster_centers = cluster_centers_normalized * std + mean

# for j in range(N):
#     filter = sample[labels==j]
#     plt.scatter(filter.iloc[:,0], filter.iloc[:,1])
# plt.show()





# %% Solve ODE at each cluster centroid and save results in PDF
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

species = [
    '$H_2$',   #0
    '$H_3^+$', #1
    '$e$',     #2
    '$He$',    #3
    '$He^+$',  #4
    '$C$',     #5
    '$CH_x$',  #6
    '$O$',     #7
    '$OH_x$',  #8
    '$CO$',    #9
    '$HCO^+$', #10
    '$C^+$',   #11
    '$M^+$',   #12
    '$M$'      #13
]

# %%
with PdfPages('centroid_plots.pdf') as pdf:
    for index, row in cluster_centers.iterrows():
        n_h = 10**row.iloc[0]
        T = 10**row.iloc[1]
        G0 = row.iloc[2]
        nelson_langer_network = build_nelson_network(n_h = n_h, T = T, G0 = G0)
        t, y = nelson_langer_network.solve_reaction([0, tf], x0)

        fig, ax = plt.subplots()
        for j in range(14):
            ax.plot(t, y[j,:], label=species[j])
        ax.set_yscale('log')
        ax.legend()
        ax.set_title(f'$n_h$: {n_h:.3f}  $T$: {T:.3f}  $G_0$: {G0:.10f}')
        pdf.savefig(fig)
        plt.close(fig)


# %% Compute the mean, std, median, max for each cluster
sample_size = 40
species_num = 9 # CO

# I need to learn how to use pandas dataframes properly... because this is messy
statistics_matrix = np.zeros(shape=(N,4))
qc_vec = np.zeros(shape=(N))
for index, row in cluster_centers.iterrows():
    # For the centroid
    n_h = 10**row.iloc[0]
    T = 10**row.iloc[1]
    G0 = row.iloc[2]
    nelson_langer_network = build_nelson_network(n_h = n_h, T = T, G0 = G0)
    t, y = nelson_langer_network.solve_reaction([0, tf], x0)
    qc_vec[index] = y[species_num,-1] # at steady state for centroid
    # Take a sample and solve the ODEs at each
    cluster_sample = params[labels==index].sample(sample_size)
    cluster_sample = cluster_sample.reset_index(drop=True)
    dvec = np.zeros(sample_size)
    for index2, row2 in cluster_sample.iterrows():
        n_h = 10**row2.iloc[0]
        T = 10**row2.iloc[1]
        G0 = row2.iloc[2]
        nelson_langer_network = build_nelson_network(n_h = n_h, T = T, G0 = G0)
        t, y = nelson_langer_network.solve_reaction([0, tf], x0)
        qj = y[species_num,-1] # at steady state for sample point near centroid
        dvec[index2] = np.abs(qj - qc_vec[index])
    # Compute statistics
    statistics_matrix[index,0] = np.mean(dvec)
    statistics_matrix[index,1] = np.median(dvec)
    statistics_matrix[index,2] = np.std(dvec)
    statistics_matrix[index,3] = np.max(dvec)

statistics = pd.DataFrame(data=np.hstack((qc_vec.reshape(-1,1), statistics_matrix)), columns=['Abundances','Mean','Median','Std','Max'])
stats_summary = pd.concat([cluster_centers, statistics], axis=1)
stats_summary['Percent Error'] = stats_summary['Max']/stats_summary['Abundances']
print(stats_summary)
stats_summary.to_csv('statistics.csv', index=False)



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

    return k, ClusterTree(left, right)



def solve_nelson_network(params_row: np.ndarray, x0: np.ndarray, QoI: int, time: float):
    n_h = 10**params_row[0]
    T = 10**params_row[1]
    G0 = params_row[2]
    network = build_nelson_network(n_h = n_h, T = T, G0 = G0)
    tvec, yvec = network.solve_reaction([0, time], x0)
    return yvec[QoI, -1]


class ClusterTree:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Cluster:
    def __init__(self, index, params, QoI_value):
        self.index = index
        self.params = params
        self.QoI_value = QoI_value

    def __str__(self):
        return f'k={self.index}\tn_h={10**self.params[0]:.3f}\tT={10**self.params[1]:.3f}\tG0={self.params[2]:.6f}'


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
msk = np.random.rand(len(params)) < 0.0001
train = params[~msk]
test = params[msk]

N_clusters, labels, tree = compute_clusters(train.reset_index(drop=True), 0.2, 9, tf, N=10)
print('Model finished training')
centroids, QoI_values = flatten_cluster_centers(tree, N_clusters)

# %% Test training data
datamat = np.empty(shape=(len(test),2)) # col1 is predicted value, col2 is actual value
k = 0
for row in test: # row is the parameters
    dvec = np.zeros(centroids.shape[0])
    j = 0
    for centroid in centroids:
        dvec[j] = np.linalg.norm(centroid, row)
    predicted_cluster = np.argmin(dvec)
    datamat[k,0] = solve_nelson_network(row, x0, 9, tf)
    datamat[k,1] = QoI_values[predicted_cluster]
    k += 1

np.save('testdata.npy', datamat)


# %% Run Kmeans for different number of clusters; save N vs silhouette score
Nvec = np.array([10, 20, 30, 40, 50])
silhouette_score_vec = np.zeros(5)
sample = params_normalized.sample(10000)
for j in range(np.size(Nvec)):
    N = Nvec[j]
    kmeans = cluster.KMeans(n_clusters=N)
    labels = kmeans.fit_predict(sample)
    silhouette_score_vec[j] = metrics.silhouette_score(sample, labels)


# %% Generate pairplot
sample['labels'] = labels
pairplot = sns.pairplot(sample, hue='labels', palette=sns.color_palette('bright'))
plt.show()

