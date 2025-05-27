# %%
import numpy as np
import pandas as pd
from nelson_langer_network import build_nelson_network
import time
from astrochem_clustering import AstrochemClusterModel

# Load parameters
datafile = np.load('data/tracer_parameter_data.npy')
datafile[:,0] *= 2
# ['nH', 'T', 'XH2', 'FUV', 'NH', 'zeta'] are the columns of datafile
data = datafile[:,[0, 1, 3]]
# data = datafile[:,[0, 1]]
data[:,[0,1]] = np.log10(data[:,[0,1]])
# data[:,2] /= 8.94e-14
params = pd.DataFrame(data=data, columns=['$\log(n_h)$', '$\log(T)$', '$G_0$'])

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


def solve_nelson_network(params_row: np.ndarray, x0: np.ndarray, QoI: int, time: float):
    n_h = 10**params_row[0]
    T = 10**params_row[1]
    G0 = params_row[2]
    network = build_nelson_network(n_h = n_h, T = T, G0 = G0)
    tvec, yvec = network.solve_reaction([0, time], x0)
    return yvec[QoI, -1]



# %% Load data

foo = np.load('data/small_dataset.npy')
small_dataset = pd.DataFrame(data=foo, columns=['$\log(n_h)$', '$\log(T)$', '$G_0$'])

np.random.seed(1234)
msk = np.random.rand(len(small_dataset)) > 0.2
train = small_dataset[msk]
test = small_dataset[~msk]

# %% Train model

start_time = time.perf_counter()
surrogate = AstrochemClusterModel()
surrogate.train_surrogate_model(train.reset_index(drop=True), 0.05, 9, x0, tf, 10, 10)
end_time = time.perf_counter()

total_time = end_time - start_time # in seconds
print(f'Training time: {total_time:.2f} seconds')


# %% If we want to save the surrogate, we need to use pickle 
import pickle
with open('data/small_dataset_model.pkl', 'wb') as file:
    pickle.dump(surrogate, file)

# %% Load the tree back in
import pickle
with open('data/small_dataset_model.pkl', 'rb') as file:
    surrogate = pickle.load(file)


# %% Test Model
start_time = time.perf_counter()
datamat = np.zeros(shape=(len(test),7))
for index, row in test.reset_index(drop=True).iterrows():
    datamat[index, 0] = solve_nelson_network(row.to_numpy(), x0, 9, tf)
mid_time = time.perf_counter()
datamat[:,[1,2]] = surrogate.predict(test.to_numpy())
end_time = time.perf_counter()
print(f'Time to solve all ODEs: {mid_time-start_time} seconds')
print(f'Time to predict data: {end_time-mid_time}')
datamat[:,3] = np.abs(datamat[:,1] - datamat[:,0]) / np.abs(datamat[:,1])
datamat[:,[4,5,6]] = test.to_numpy()

# %% Error statistics
print(f'Mean: {np.mean(datamat[:,3])}')
print(f'Median: {np.median(datamat[:,3])}')
print(f'Max: {np.max(datamat[:,3])}')
print(f'STD: {np.std(datamat[:,3])}')
num_vals_outside_error = (datamat[:,3] > 0.05).sum()
print(f'# of points outside error: {num_vals_outside_error} ({num_vals_outside_error/len(datamat[:,2])}% of data)')


# %% debug problem points
problem_points = datamat[datamat[:,3] > 0.1]
expected_clusters = surrogate.predict_with_loop(datamat[:,[4,5,6]])
predicted_clusters = surrogate.predict(datamat[:,[4,5,6]])










# %% Visualization of clusters
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
