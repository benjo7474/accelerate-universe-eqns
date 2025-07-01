#### This is the code that I am using for essentially everything
#### At some point need to turn this into non-spaghetti code
# Or just say this has the form of a notebook. Maybe I should turn this into a jupyter notebook...


# %%
import numpy as np
import pandas as pd
from nelson_langer_network import build_nelson_network
import time
import torch
from astrochem_clustering import AstrochemClusterModel
import matplotlib.pyplot as plt

# Load parameters

def load_parameters():
    datafile = np.load('data/tracer_parameter_data.npy')
    datafile[:,0] *= 2
    # ['nH', 'T', 'XH2', 'FUV', 'NH', 'zeta'] are the columns of datafile
    data = datafile[:,[0, 1, 3]]
    # data = datafile[:,[0, 1]]
    data[:,[0,1]] = np.log10(data[:,[0,1]])
    # data[:,2] /= 8.94e-14
    params = pd.DataFrame(data=data, columns=['$\log(n_h)$', '$\log(T)$', '$G_0$'])
    return params

# Data for ode solves
# [H_2, H_3^+, e, He, He^+, C, CH_x, O, OH_x, CO, HCO^+, C^+, M^+, M]
#  0    1      2  3   4     5  6     7  8     9   10     11   12   13

# These are the initial conditions used by Nina in her testing code
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
# From the overleaf doc
# x0 = np.array([
    # what should go here anyways?
# ])

secs_per_year = 3600*24*365
years = 10000
tf = years * secs_per_year


def solve_nelson_network(params_row: np.ndarray, x0: np.ndarray, QoI: np.ndarray, time: float, teval=None):
    n_h = 10**params_row[0]
    T = 10**params_row[1]
    G0 = params_row[2]
    p = torch.tensor([n_h, T, G0])
    network = build_nelson_network(params=p, compute_sensitivities=False)
    tvec, yvec = network.solve_reaction([0, time], x0, teval)
    if teval == None:
        return yvec[QoI, -1]
    else:
        return yvec[QoI, :]



# %% Load small dataset

def load_small_dataset():
    foo = np.load('data/small_dataset.npy')
    small_dataset = pd.DataFrame(data=foo, columns=['$\log(n_h)$', '$\log(T)$', '$G_0$'])
    np.random.seed(1234)
    msk = np.random.rand(len(small_dataset)) > 0.2
    train = small_dataset[msk]
    test = small_dataset[~msk]
    return train, test


# %% Load medium dataset (with ode solves)
# return training and testing data pandas dfs (should put seed as a parameter)

def load_medium_dataset_with_ode_solves():
    foo = np.load('data/medium_dataset_with_ode_solves.npy')
    medium_dataset = pd.DataFrame(data=foo, columns=[
        '$\log(n_h)$',
        '$\log(T)$',
        '$G_0$',
        'e (t=100Y)',
        'H_2 (t=100Y)',
        'H_3+ (t=100Y)',
        'CO (t=100Y)',
        'C (t=100Y)',
        'e (t=1000Y)',
        'H_2 (t=1000Y)',
        'H_3+ (t=1000Y)',
        'CO (t=1000Y)',
        'C (t=1000Y)',
        'e (t=10000Y)',
        'H_2 (t=10000Y)',
        'H_3+ (t=10000Y)',
        'CO (t=10000Y)',
        'C (t=10000Y)',
        'e (t=40000Y)',
        'H_2 (t=40000Y)',
        'H_3+ (t=40000Y)',
        'CO (t=40000Y)',
        'C (t=40000Y)'
    ])
    np.random.seed(1234)
    msk = np.random.rand(len(medium_dataset)) > 0.2
    train = medium_dataset[msk]
    test = medium_dataset[~msk]
    return train, test



# %% Load large dataset (with ode solves)

def load_large_dataset_with_ode_solves():
    foo = np.load('data/large_dataset_with_ode_solves.npy')
    large_dataset = pd.DataFrame(data=foo, columns=[
        '$\log(n_h)$',
        '$\log(T)$',
        '$G_0$',
        'e (t=100Y)',
        'H_2 (t=100Y)',
        'H_3+ (t=100Y)',
        'CO (t=100Y)',
        'C (t=100Y)',
        'e (t=1000Y)',
        'H_2 (t=1000Y)',
        'H_3+ (t=1000Y)',
        'CO (t=1000Y)',
        'C (t=1000Y)',
        'e (t=10000Y)',
        'H_2 (t=10000Y)',
        'H_3+ (t=10000Y)',
        'CO (t=10000Y)',
        'C (t=10000Y)',
        'e (t=40000Y)',
        'H_2 (t=40000Y)',
        'H_3+ (t=40000Y)',
        'CO (t=40000Y)',
        'C (t=40000Y)'
    ])
    train = large_dataset.loc[:99999]
    test = large_dataset.loc[100000:]
    return train, test


# %% Load FULL dataset (with ODE solves!)

def load_full_dataset_with_ode_solves():
    foo = np.load('data/tracer_parameter_data_with_ode_solves.npy')
    full_dataset = pd.DataFrame(data=foo, columns=[
        '$\log(n_h)$',
        '$\log(T)$',
        '$G_0$',
        'e (t=100Y)',
        'H_2 (t=100Y)',
        'H_3+ (t=100Y)',
        'CO (t=100Y)',
        'C (t=100Y)',
        'e (t=1000Y)',
        'H_2 (t=1000Y)',
        'H_3+ (t=1000Y)',
        'CO (t=1000Y)',
        'C (t=1000Y)',
        'e (t=10000Y)',
        'H_2 (t=10000Y)',
        'H_3+ (t=10000Y)',
        'CO (t=10000Y)',
        'C (t=10000Y)',
        'e (t=40000Y)',
        'H_2 (t=40000Y)',
        'H_3+ (t=40000Y)',
        'CO (t=40000Y)',
        'C (t=40000Y)'
    ])
    np.random.seed(1234)
    msk = np.random.rand(len(full_dataset)) > 0.2
    train = full_dataset[msk]
    test = full_dataset[~msk]
    return train, test



# %% Better sampling algorithm (as suggested by George)
# Run FAISS for the entire 2M dataset

def better_sampling_algorithm():
    params = load_parameters()
    import faiss
    faiss_index_params = faiss.IndexFlatL2(3)
    faiss_index_params.add(params.to_numpy())
    D, I = faiss_index_params.search(params.to_numpy(), 20) # this takes a long time
    # I is saved in data/tracer_param_data_nearest_ind.npy

    from numba import jit
    @jit(nopython=True)
    def count_entries(I):
        occurences = np.zeros(shape=len(I))
        for ind, x in np.ndenumerate(I):
            occurences[x] += 1
        return np.argsort()

    count_entries(I[:100,:])



# %% Train model

def train_model(train_data):
    start_time = time.perf_counter()
    surrogate = AstrochemClusterModel()
    surrogate.train_surrogate_model(train_data.reset_index(drop=True), 0.01, [2, 0, 1, 9, 5], x0, tf, 10, 10, False, [8,9,10,11,12])
    end_time = time.perf_counter()

    total_time = end_time - start_time # in seconds
    print(f'Training time: {total_time:.2f} seconds')


# %% Bonus test case where we use the MAX number of clusters;
#    i.e. each training data point as a cluster centroid (target point)
def train_model_max_and_get_datamatrix(train, test):
    import faiss
    centroids = train.to_numpy()[:,[0,1,2]]
    QoI_values_ind = [13,14,15,16,17]
    QoI_values = train.to_numpy()[:,QoI_values_ind]
    faiss_index = faiss.IndexFlatL2(3)
    faiss_index.add(centroids)

    start_time = time.perf_counter()
    D, I = faiss_index.search(test.to_numpy()[:,[0,1,2]], 1)
    end_time = time.perf_counter()
    total_time = end_time - start_time # in seconds
    print(f'Time: {total_time:.2f} seconds')

    datamat = np.zeros(shape=(len(test),24))
    datamat[:,[0,1,2,3,4]] = test.to_numpy()[:,QoI_values_ind] # exact (already solved)
    datamat[:,5] = I[:,0]
    datamat[:,[6,7,8,9,10]] = QoI_values[I[:,0]] # predicted values
    # percent errors
    datamat[:,[11,12,13,14,15]] = np.abs(datamat[:,[6,7,8,9,10]] - datamat[:,[0,1,2,3,4]]) / np.abs(datamat[:,[0,1,2,3,4]])
    # relative errors
    datamat[:,[16,17,18,19,20]] = np.abs(datamat[:,[6,7,8,9,10]] - datamat[:,[0,1,2,3,4]])
    # test data
    datamat[:,[21,22,23]] = test.to_numpy()[:,[0,1,2]]

    return datamat

# %% If we want to save the surrogate, we need to use pickle 
def save_surrogate(surrogate):
    import pickle
    with open(f'data/medium_model_tf_{tf}_tol_1.0.pkl', 'wb') as file:
        pickle.dump(surrogate, file)

# %% Load the tree back in with pickle
def load_surrogate(filename):
    import pickle
    with open(filename, 'rb') as file:
        surrogate = pickle.load(file)
        return surrogate


# %% Test Model
def get_datamatrix(dataset, surrogate, test):
    start_time = time.perf_counter()
    datamat = np.zeros(shape=(len(test),24))
    # Exact solution for e, H2, H3+, CO, C in columns 0-4
    # Index in column 5
    # Predictions in columns 6-10
    # Relative errors in columns 11-15
    # Absolute errors in columns 16-20
    # Test data in columns 21-23
    j = 0
    for index, row in test.iterrows():
        datamat[j, [0,1,2,3,4]] = dataset.iloc[index, [8,9,10,11,12]].to_numpy()
        # datamat[j, [0,1,2,3,4]] = solve_nelson_network(row.to_numpy()[0:3], x0, [2,0,1,9,5], tf)
        j += 1
    mid_time = time.perf_counter()
    datamat[:,[5,6,7,8,9,10]] = surrogate.predict(test[['$\log(n_h)$','$\log(T)$','$G_0$']].to_numpy())
    end_time = time.perf_counter()
    print(f'Time to solve all ODEs: {mid_time-start_time} seconds')
    print(f'Time to predict data: {end_time-mid_time}')
    datamat[:,[11,12,13,14,15]] = np.abs(datamat[:,[6,7,8,9,10]] - datamat[:,[0,1,2,3,4]]) / np.abs(datamat[:,[0,1,2,3,4]])
    datamat[:,[16,17,18,19,20]] = np.abs(datamat[:,[6,7,8,9,10]] - datamat[:,[0,1,2,3,4]])
    datamat[:,[21,22,23]] = test[['$\log(n_h)$','$\log(T)$','$G_0$']].to_numpy()



def datamatrix_to_pandas(datamat):
    pandas_datamat = pd.DataFrame(data=datamat, columns=[
        'Exact e',              # 0
        'Exact H2',             # 1
        'Exact H3+',            # 2
        'Exact CO',             # 3
        'Exact C',              # 4
        'Closest Index',        # 5
        'Predicted e',          # 6
        'Predicted H2',         # 7
        'Predicted H3+',        # 8
        'Predicted CO',         # 9
        'Predicted C',          # 10
        'Percent Error e',      # 11
        'Percent Error H2',     # 12
        'Percent Error H3+',    # 13
        'Percent Error CO',     # 14
        'Percent Error C',      # 15
        'Absolute Error e',     # 16
        'Absolute Error H2',    # 17
        'Absolute Error H3+',   # 18
        'Absolute Error CO',    # 19
        'Absolute Error C',     # 20
        '$\log(n_h)$',          # 21
        '$\log(T)$',            # 22
        '$G_0$'                 # 23
    ])
    return pandas_datamat







######## ------ PLOTS AND ERROR ANALYSIS ------ ########




# %% Error statistics

def print_error_statistics(datamat):
    ind = 11
    print('Stats for e (10K years)')
    print(f'Mean: {np.mean(datamat[:,ind])}')
    print(f'Median: {np.median(datamat[:,ind])}')
    print(f'Max: {np.max(datamat[:,ind])}')
    print(f'STD: {np.std(datamat[:,ind])}')
    tol = 0.1
    num_vals_outside_error = (datamat[:,ind] > tol).sum()
    print(f'# of points outside {tol*100}% error: {num_vals_outside_error} ({num_vals_outside_error/len(datamat[:,2])}% of data)\n')

    ind = 12
    print('Stats for H2 (1K years)')
    print(f'Mean: {np.mean(datamat[:,ind])}')
    print(f'Median: {np.median(datamat[:,ind])}')
    print(f'Max: {np.max(datamat[:,ind])}')
    print(f'STD: {np.std(datamat[:,ind])}')
    num_vals_outside_error = (datamat[:,ind] > tol).sum()
    print(f'# of points outside {tol*100}% error: {num_vals_outside_error} ({num_vals_outside_error/len(datamat[:,2])}% of data)\n')

    ind = 13
    print('Stats for H3+ (1K years)')
    print(f'Mean: {np.mean(datamat[:,ind])}')
    print(f'Median: {np.median(datamat[:,ind])}')
    print(f'Max: {np.max(datamat[:,ind])}')
    print(f'STD: {np.std(datamat[:,ind])}')
    num_vals_outside_error = (datamat[:,ind] > tol).sum()
    print(f'# of points outside {tol*100}% error: {num_vals_outside_error} ({num_vals_outside_error/len(datamat[:,2])}% of data)\n')

    ind = 14
    print('Stats for CO (1K years)')
    print(f'Mean: {np.mean(datamat[:,ind])}')
    print(f'Median: {np.median(datamat[:,ind])}')
    print(f'Max: {np.max(datamat[:,ind])}')
    print(f'STD: {np.std(datamat[:,ind])}')
    num_vals_outside_error = (datamat[:,ind] > tol).sum()
    print(f'# of points outside {tol*100}% error: {num_vals_outside_error} ({num_vals_outside_error/len(datamat[:,2])}% of data)\n')

    ind = 15
    print('Stats for C (1K years)')
    print(f'Mean: {np.mean(datamat[:,ind])}')
    print(f'Median: {np.median(datamat[:,ind])}')
    print(f'Max: {np.max(datamat[:,ind])}')
    print(f'STD: {np.std(datamat[:,ind])}')
    num_vals_outside_error = (datamat[:,ind] > tol).sum()
    print(f'# of points outside {tol*100}% error: {num_vals_outside_error} ({num_vals_outside_error/len(datamat[:,2])}% of data)')



# %% relative errors

def plot_all_relative_errors(datamat):
    plt.figure()
    plt.title('Percent Error: e (40K years)')
    plt.hist(datamat[:,11], bins=np.arange(0, 0.3, 0.005))
    plt.figure()
    plt.title('Percent Error: H2 (40K years)')
    plt.hist(datamat[:,12], bins=np.arange(0, 0.3, 0.005))
    plt.figure()
    plt.title('Percent Error: H3+ (40K years)')
    plt.hist(datamat[:,13], bins=np.arange(0, 0.3, 0.005))
    plt.figure()
    plt.title('Percent Error: CO (40K years)')
    plt.hist(datamat[:,14], bins=np.arange(0, 0.3, 0.005))
    plt.figure()
    plt.title('Percent Error: C (40K years)')
    plt.hist(datamat[:,15], bins=np.arange(0, 0.3, 0.005))

# %% absolute errors (log)

def plot_all_absolute_errors(datamat):
    plt.figure()
    plt.title('Absolute Error: e (40K years)')
    plt.hist(np.log10(datamat[:,16]))
    plt.figure()
    plt.title('Absolute Error: H2 (40K years)')
    plt.hist(np.log10(datamat[:,17]))
    plt.figure()
    plt.title('Absolute Error: H3+ (40K years)')
    plt.hist(np.log10(datamat[:,18]))
    plt.figure()
    plt.title('Absolute Error: CO (40K years)')
    plt.hist(np.log10(datamat[:,19]))
    plt.figure()
    plt.title('Absolute Error: C (40K years)')
    plt.hist(np.log10(datamat[:,20]))


# %% 2-d histograms (relative vs absolute errors)
def plot_relative_vs_absolute_error(datamat):
    from matplotlib.colors import ListedColormap, BoundaryNorm

    boundaries = [-1, 1e-16, 10, np.inf]
    colors = ['white','red', 'green']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    plt.figure()
    plt.title('Relative Error vs. Absolute Error: e (10K years)')
    plt.hist2d(datamat[:,11], np.log10(datamat[:,16]), range=[[0,0.1],[-16,-4]], bins=[50,50], cmap=cmap, norm=norm)
    plt.xlabel('Relative Error')
    plt.ylabel('Absolute Error (log)')

    plt.figure()
    plt.title('Relative Error vs. Absolute Error: H2 (10K years)')
    plt.hist2d(datamat[:,12], np.log10(datamat[:,17]), range=[[0,0.1],[-16,-4]], bins=[50,50], cmap=cmap, norm=norm)
    plt.xlabel('Relative Error')
    plt.ylabel('Absolute Error (log)')

    plt.figure()
    plt.title('Relative Error vs. Absolute Error: H3+ (10K years)')
    plt.hist2d(datamat[:,13], np.log10(datamat[:,18]), range=[[0,0.2],[-16,-4]], bins=[50,50], cmap=cmap, norm=norm)
    plt.xlabel('Relative Error')
    plt.ylabel('Absolute Error (log)')

    plt.figure()
    plt.title('Relative Error vs. Absolute Error: CO (10K years)')
    plt.hist2d(datamat[:,14], np.log10(datamat[:,19]), range=[[0,0.2],[-16,-4]], bins=[50,50], cmap=cmap, norm=norm)
    plt.xlabel('Relative Error')
    plt.ylabel('Absolute Error (log)')

    plt.figure()
    plt.title('Relative Error vs. Absolute Error: C (10K years)')
    plt.hist2d(datamat[:,15], np.log10(datamat[:,20]), range=[[0,0.1],[-16,-4]], bins=[50,50], cmap=cmap, norm=norm)
    plt.xlabel('Relative Error')
    plt.ylabel('Absolute Error (log)')

# %% ground truth vs relative error
def plot_ground_truth_vs_relative_error(datamat):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    boundaries = [-1, 1e-16, 10, np.inf]
    colors = ['white','red', 'green']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)
    plt.figure()
    plt.title('Ground Truth vs. Relative Error: CO (10K years)')
    plt.hist2d(np.log10(datamat[:,3]), datamat[:,14], range=[[-16,-4],[0,0.2]], bins=[50,50], cmap=cmap, norm=norm)
    plt.xlabel('Ground Truth of CO (log)')
    plt.ylabel('Relative Error')


# %% filter out "bad" points (log(CO) > -6 and rel err > 0.1) and plot distances on scatter plot
def plot_bad_points(pandas_datamat, train):
    bad_points = pandas_datamat[pandas_datamat['Exact CO'] > 1e-6]
    bad_points_closest_params = train.iloc[bad_points['Closest Index'].to_numpy()]
    log_nh_distances = np.abs(bad_points_closest_params['$\log(n_h)$'].to_numpy() - bad_points['$\log(n_h)$'].to_numpy())
    log_T_distances = np.abs(bad_points_closest_params['$\log(T)$'] - bad_points['$\log(T)$'].to_numpy())
    G0_distances = np.abs(bad_points_closest_params['$G_0$'] - bad_points['$G_0$'].to_numpy())
    plt.scatter(log_nh_distances, log_T_distances, G0_distances)
    plt.xlabel('$\\log(n_h)$ distances')
    plt.ylabel('$\\log(T)$ distances')
    plt.xlim(0,0.2)
    plt.ylim(0,0.2)



# %% Visualization of clusters

def cluster_visualization(params, surrogate):
    import matplotlib.pyplot as plt

    plt.subplot(2,2,1)
    plt.scatter(params['$\log(n_h)$'],params['$\log(T)$'],marker='x',s=0.5)
    plt.xlabel('$\log(n_h)$')
    plt.ylabel('$\log(T)$')

    plt.subplot(2,2,2)
    N_centroids = 10 * 2**0
    current_centroids_0 = np.zeros(shape=(N_centroids, 3))
    j = 0
    for i in range(N_centroids):
        current_centroids_0[j,:] = surrogate.tree[i].params
        j += 1

    current_centroids_0 = current_centroids_0 * surrogate.std.to_numpy() + surrogate.mean.to_numpy()
    plt.scatter(params['$\log(n_h)$'],params['$\log(T)$'],marker='x',s=0.5)
    plt.scatter(current_centroids_0[:,0], current_centroids_0[:,1], s=8)
    plt.xlabel('$\log(n_h)$')
    plt.ylabel('$\log(T)$')

    plt.subplot(2,2,3)
    N_centroids = 10 * 2**1
    current_centroids_1 = np.zeros(shape=(N_centroids, 3))
    j = 0
    for i in range(10):
        current_centroids_1[j,:] = surrogate.tree[i].left.params
        j += 1
        current_centroids_1[j,:] = surrogate.tree[i].right.params
        j += 1

    current_centroids_1 = current_centroids_1 * surrogate.std.to_numpy() + surrogate.mean.to_numpy()
    plt.scatter(params['$\log(n_h)$'],params['$\log(T)$'],marker='x',s=0.5)
    plt.scatter(current_centroids_0[:,0], current_centroids_0[:,1], s=8)
    plt.scatter(current_centroids_1[:,0], current_centroids_1[:,1], s=8)
    plt.xlabel('$\log(n_h)$')
    plt.ylabel('$\log(T)$')


    plt.subplot(2,2,4)
    N_centroids = 10 * 2**2
    current_centroids_2 = np.zeros(shape=(N_centroids, 3))
    j = 0
    for i in range(10):
        current_centroids_2[j,:] = surrogate.tree[i].left.left.params
        j += 1
        current_centroids_2[j,:] = surrogate.tree[i].left.right.params
        j += 1
        current_centroids_2[j,:] = surrogate.tree[i].right.left.params
        j += 1
        current_centroids_2[j,:] = surrogate.tree[i].right.right.params
        j += 1
        

    current_centroids_2 = current_centroids_2 * surrogate.std.to_numpy() + surrogate.mean.to_numpy()
    plt.scatter(params['$\log(n_h)$'],params['$\log(T)$'],marker='x',s=0.5)
    plt.scatter(current_centroids_0[:,0], current_centroids_0[:,1], s=8)
    plt.scatter(current_centroids_1[:,0], current_centroids_1[:,1], s=8)
    plt.scatter(current_centroids_2[:,0], current_centroids_2[:,1], s=8)
    plt.xlabel('$\log(n_h)$')
    plt.ylabel('$\log(T)$')
    plt.show()

# %%
