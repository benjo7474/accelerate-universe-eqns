# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import torch
from nelson_langer_network import solve_nelson_network
from matplotlib.backends.backend_pdf import PdfPages


x0 = np.array([
    0.5,      # H_2
    9.059e-9, # H_3^+
    2e-4,     # e
    0.1,      # He
    7.866e-7, # He^+
    0.0,      # C
    0.0,      # CH_x
    0.0004,   # O
    0.0,      # OH_x
    0.0,      # CO
    0.0,      # HCO^+
    0.0002,   # C^+
    2.0e-7,   # M^+
    2.0e-7    # M
])

# ICs that I partly took from overleaf, partly made up
# x0 = np.array([
#     0.5,      # H_2
#     0.25,     # H_3^+
#     5.324e-6, # e
#     0.1,      # He
#     7.866e-7, # He^+
#     1.77e-4,  # C
#     0.0,      # CH_x
#     0.0004,   # O
#     0.0,      # OH_x
#     0.0,      # CO
#     0.0,      # HCO^+
#     1e-10,    # C^+
#     2.0e-7,   # M^+
#     2.0e-7    # M
# ])

secs_per_year = 3600*24*365
tf = 10000 * secs_per_year

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

QoI = 9


def load_parameters():
    params = np.load('TACC_code/input_p_full.npy')
    params[:,[0,1]] = np.log10(params[:,[0,1]])
    return params

def load_parameters_with_solves(solves_filename = 'TACC_code/output_q_CO_full.npy'):
    params = np.load('TACC_code/input_p_full.npy')
    params[:,[0,1]] = np.log10(params[:,[0,1]])
    outputs = np.load(solves_filename)
    return params, outputs[:,2]



def cluster_params(params: np.ndarray, N=20, normalize=False, plot_centroids=True, solve_odes_centroids=False):

    if normalize == True:
        mean = params.mean(axis=0)
        std = params.std(axis=0)
        params_normalized = (params - mean) / std
        kmeans = cluster.KMeans(n_clusters=N, random_state=1)
        labels = kmeans.fit_predict(params_normalized)
        cluster_centers = kmeans.cluster_centers_ * std + mean
    elif normalize == False:
        kmeans = cluster.KMeans(n_clusters=N, random_state=1)
        labels = kmeans.fit_predict(params[:,[0,1]])
        cluster_centers = kmeans.cluster_centers_

    if plot_centroids == True:
        for j in range(N):
            filter = params[labels==j]
            plt.scatter(filter[:,0], filter[:,1], s=2)
        plt.scatter(cluster_centers[:,0], cluster_centers[:,1], c='black', s=20, label='Centroids')
        plt.title(f'Clustering of $\log(n_h)$ and $\log(T)$ ($k={N}$)')
        plt.legend(loc='upper right')
        plt.xlabel('$\log(n_h)$')
        plt.ylabel('$\log(T)$')
        plt.show()

    if solve_odes_centroids == True:
        solve_odes_at_centroids(cluster_centers)

    return cluster_centers, labels



def solve_odes_at_centroids(centroids, pdf_name = 'testing.pdf'):

    with PdfPages(pdf_name) as pdf:
        for index, row in enumerate(centroids):
            t, y = solve_nelson_network(row, x0, np.arange(14), tf)

            fig, ax = plt.subplots()
            for j in range(14):
                ax.plot(t, y[j,:], label=species[j])
            ax.set_yscale('log')
            ax.legend()
            ax.set_title(f'$n_h$: {10**row[0]:.3f}  $T$: {10**row[1]:.3f}  $G_0$: {row[2]:.10f}')
            pdf.savefig(fig)
            plt.close(fig)



def compute_statistics(params, solves, N=100, QoI=9, plot_percent_error=True):

    cluster_centers, labels = cluster_params(params, N, normalize=True, plot_centroids=False)
    N = len(cluster_centers)
    statistics_matrix = np.zeros(shape=(N,4))
    qc_vec = np.zeros(N)
    for index, row in enumerate(cluster_centers):
        # For the centroid
        params_in_cluster = params[labels==index]
        solves_in_cluster = solves[labels==index]
        qc_vec[index] = solve_nelson_network(row, x0, QoI, tf)
        p_error = np.abs(qc_vec[index] - solves_in_cluster)
        # Compute statistics
        statistics_matrix[index,0] = np.mean(p_error)
        statistics_matrix[index,1] = np.median(p_error)
        statistics_matrix[index,2] = np.std(p_error)
        statistics_matrix[index,3] = np.max(p_error)

    statistics = pd.DataFrame(data=np.hstack((cluster_centers, qc_vec.reshape(-1,1), statistics_matrix)),
                              columns=['$\log(n_h)$','$\log(T)$','G_0','Abundances','Mean','Median','Std','Max'])
    statistics['Percent Error'] = statistics['Max']/statistics['Abundances']

    if plot_percent_error == True:
        print('foo')

    return statistics



def compute_silhouette_scores(params, Nvec = np.array([10, 20, 30, 40, 50]), ss=10000):
    silhouette_score_vec = np.zeros(len(Nvec))
    sample = np.random.choice(params, ss, replace=False)
    for j in range(len(Nvec)):
        N = Nvec[j]
        kmeans = cluster.KMeans(n_clusters=N)
        labels = kmeans.fit_predict(sample)
        silhouette_score_vec[j] = metrics.silhouette_score(sample, labels)

    


if __name__ == '__main__':
    params, solves = load_parameters_with_solves()
    # centroids, labels = cluster_params(params, N=200, normalize=True, solve_odes_centroids=False)
    statistics = compute_statistics(params, solves, 5000, QoI, plot_percent_error=True)
    


# %%
