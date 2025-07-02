# Represents a cluster model for astro parameters.
# The class AstrochemClusterModel, after calling the function train_surrogate_model(),
# contains cluster centroids and QoI evaluations at those centroids that we can use
# k-nearest neighbors to obtain a good approximation.


import numpy as np
import pandas as pd
import sklearn.cluster
from nelson_langer_network import build_nelson_network
import faiss


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


class AstrochemClusterModel:

    def __init__(self):
        self.N_clusters = 0
        self.model_trained = False


    def train_surrogate_model_max_targets(self,
        training_data, # NOT normalized, pandas dataframe
        QoI: int,
        x0: np.ndarray,
        time: float,
        ode_solve_indices: np.ndarray = None # if not None, uses these columns in training_data for the solves
        ):

        # still need to normalize to make predict work for both training functions
        first_columns = training_data[['$\log(n_h)$', '$\log(T)$', '$G_0$']]
        self.mean = first_columns.mean()
        self.std = first_columns.std()
        first_columns = (first_columns - self.mean) / self.std
        training_data[['$\log(n_h)$', '$\log(T)$', '$G_0$']] = first_columns

        self.QoI_length = len(QoI)
        centroids = training_data.to_numpy()[:,[0,1,2]]
        self.faiss_index = faiss.IndexFlatL2(3)
        self.faiss_index.add(centroids)
        if ode_solve_indices == None:
            # we need to do every solve. bummer.
            self.QoI_values = np.zeros((len(training_data), len(QoI)))
            for j, params_row in enumerate(training_data.to_numpy()):
                self.QoI_values[j,:] = self._solve_nelson_network(params_row, x0, QoI, time)
        else:
            self.QoI_values = training_data.to_numpy()[:,ode_solve_indices]

        return len(training_data)




    def train_surrogate_model(self,
        training_data, # NOT normalized, pandas dataframe
        error_tol: float,
        QoI: int,
        x0: np.ndarray, # initial condition for ode solves
        time: float, # final time for ode solves
        N: int = 10, # number of initial clusters to split into
        ss: int = 40, # sample size used to compute statistics in each cluster
        ode_solve_indices: np.ndarray = None, # if this is not none, uses these columns in training_data for the solves
        ):
        # Computes the clusters recursively, and makes smaller based on error_tol.

        # first process the parameters; i.e. normalize the first three columns (those used to cluster)
        first_columns = training_data[['$\log(n_h)$', '$\log(T)$', '$G_0$']]
        self.mean = first_columns.mean()
        self.std = first_columns.std()
        first_columns = (first_columns - self.mean) / self.std
        training_data[['$\log(n_h)$', '$\log(T)$', '$G_0$']] = first_columns
        self.QoI_length = len(QoI)

        k = 0 # counting number of clusters as we go
        indices = np.full(shape=len(training_data), fill_value=-1)

        kmeans = sklearn.cluster.KMeans(n_clusters=N, random_state=1)
        labels = kmeans.fit_predict(first_columns)
        centroids = kmeans.cluster_centers_
        cluster_structure = np.empty(shape=N, dtype=object)

        for j in range(N):

            # Compute error inside each cluster
            params_in_cluster = training_data[labels==j]

            ss_new = ss
            if len(params_in_cluster) < ss:
                ss_new = len(params_in_cluster)
            
            centroid_params = centroids[j]
            qc = self._solve_nelson_network(centroid_params, x0, QoI, time)

            # Compute error via max_{j=1,...,ss}(q_c - q_j) / q_c
            # param_sample = params_in_cluster.sample(ss_new)
            # dvec = np.empty(shape=ss_new)
            # for i, row in param_sample.reset_index(drop=True).iterrows():
            #     qi = self._solve_nelson_network(row.to_numpy(), x0, QoI, time)
            #     dvec[i] = np.abs(qc-qi)
            # error = np.max(dvec) / qc

            # Compute error by taking largest distances away and averaging errors
            distances_from_cluster = np.linalg.norm(centroid_params - params_in_cluster[['$\log(n_h)$','$\log(T)$','$G_0$']].to_numpy(), axis=1)
            largest_row_numbers = np.argpartition(distances_from_cluster, -ss_new)[-ss_new:]
            dvec = np.empty(shape=(ss_new, len(QoI)))
            for i, row in params_in_cluster.iloc[largest_row_numbers].reset_index(drop=True).iterrows():
                if ode_solve_indices == None:
                    qi = self._solve_nelson_network(row.to_numpy()[0:3], x0, QoI, time)
                else:
                    qi = row.to_numpy()[ode_solve_indices]
                dvec[i,:] = np.abs(qc-qi)
            error = np.max(dvec / qc)

            if error > error_tol:
                # Split cluster in half (recursive step)
                k, cluster_structure[j] = self._compute_clusters_recursion_helper(
                        params_in_cluster, centroid_params, error_tol, QoI, x0, time,
                        indices, ss=ss, ode_solve_indices=ode_solve_indices, k=k)
            else:
                # Tolerance is good; save values in cluster column
                indices[params_in_cluster.index.to_numpy()] = k
                cluster_structure[j] = Cluster(k, centroid_params, qc)
                k += 1

        self.N_clusters = k
        self.model_trained = True

        # save the tree structure.
        # first flatten
        all_centroids, self.QoI_values = self._flatten_cluster_centers(cluster_structure)
        # put this into faiss
        self.faiss_index = faiss.IndexFlatL2(3)
        self.faiss_index.add(all_centroids)

        return k


    def _compute_clusters_recursion_helper(self,
        params: pd.DataFrame,
        prev_centroid: np.ndarray,
        error_tol: float,
        QoI: int,
        x0: np.ndarray,
        time: float,
        indices: np.ndarray, # array containing cluster labels (reference)
        ss: int = 40, # sample size used to compute statistics in each cluster
        ode_solve_indices: np.ndarray = None,
        k: int = 0 # current cluster index (for recursive purposes)
        ):

        N = 2  # split in half
        kmeans = sklearn.cluster.KMeans(n_clusters=N)
        cluster_columns = params[['$\log(n_h)$', '$\log(T)$', '$G_0$']]
        labels = kmeans.fit_predict(cluster_columns)
        centroids = kmeans.cluster_centers_

        # First cluster (N=0)
        params_in_cluster_0 = params[labels==0]

        ss_new = ss
        if len(params_in_cluster_0) < ss:
            # Cluster error is not converging in this case. This is a big issue
            # Can alternatively just use this as a cluster
            ss_new = len(params_in_cluster_0)

        centroid_params = centroids[0]
        qc = self._solve_nelson_network(centroid_params, x0, QoI, time)

        # Compute error by taking largest distances away and averaging errors
        # can make this block wayyyy faster using either faiss or matrices
        distances_from_cluster = np.linalg.norm(centroid_params - params_in_cluster_0[['$\log(n_h)$','$\log(T)$','$G_0$']].to_numpy(), axis=1)
        largest_row_numbers = np.argpartition(distances_from_cluster, -ss_new)[-ss_new:]
        dvec = np.empty(shape=(ss_new, len(QoI)))
        for i, row in params_in_cluster_0.iloc[largest_row_numbers].reset_index(drop=True).iterrows():
            if ode_solve_indices == None:
                qi = self._solve_nelson_network(row.to_numpy()[0:3], x0, QoI, time)
            else:
                qi = row.to_numpy()[ode_solve_indices]
            dvec[i,:] = np.abs(qc-qi)
        error = np.max(dvec / qc)

        if error > error_tol:
            # Split cluster in half (recursive step)
            k, left = self._compute_clusters_recursion_helper(
                    params_in_cluster_0, centroid_params, error_tol, QoI, x0, time, indices, ss=ss, k=k)
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
        
        centroid_params = centroids[1]
        qc = self._solve_nelson_network(centroid_params, x0, QoI, time)

        # Compute error by taking largest distances away and averaging errors
        # can make this block wayyyy faster using either faiss or matrices
        distances_from_cluster = np.linalg.norm(centroid_params - params_in_cluster_1[['$\log(n_h)$','$\log(T)$','$G_0$']].to_numpy(), axis=1)
        largest_row_numbers = np.argpartition(distances_from_cluster, -ss_new)[-ss_new:]
        dvec = np.empty(shape=(ss_new, len(QoI)))
        for i, row in params_in_cluster_1.iloc[largest_row_numbers].reset_index(drop=True).iterrows():
            if ode_solve_indices == None:
                qi = self._solve_nelson_network(row.to_numpy()[0:3], x0, QoI, time)
            else:
                qi = row.to_numpy()[ode_solve_indices]
            dvec[i,:] = np.abs(qc-qi)
        error = np.max(dvec / qc)

        if (error > error_tol).any():
            # Split cluster in half (recursive step)
            k, right = self._compute_clusters_recursion_helper(
                    params_in_cluster_1, centroid_params, error_tol, QoI, x0, time, indices, ss=ss, k=k)
        else:
            # Tolerance is good; save values in cluster column
            indices[params_in_cluster_1.index.to_numpy()] = k
            right = Cluster(k, centroid_params, qc)
            k += 1

        return k, ClusterTree(prev_centroid, left, right)



    # method takes in NORMALIZED parameters via numpy
    def _solve_nelson_network(self, params_row: np.ndarray, x0: np.ndarray, QoI: int, time: float, QoI_derivative=None):
        n_h = 10**(self.std.iloc[0] * params_row[0] + self.mean.iloc[0])
        T = 10**(self.std.iloc[1] * params_row[1] + self.mean.iloc[1])
        G0 = self.std.iloc[2] * params_row[2] + self.mean.iloc[2]
        if QoI_derivative == None:
            network = build_nelson_network(params=np.array([n_h, T, G0]), compute_sensitivities=False)
            _, yvec = network.solve_reaction([0, time], x0, teval=[time])
        return yvec.flatten()[QoI]

    

    def _flatten_cluster_centers(self, tree):
        centroids = np.zeros(shape=(self.N_clusters, 3))
        QoI_values = np.zeros(shape=(self.N_clusters, self.QoI_length))
        k = 0
        for val in tree:
            if type(val) == Cluster:
                centroids[k] = val.params
                QoI_values[k,:] = val.QoI_value
                k += 1
            elif type(val) == ClusterTree:
                k = self._flatten_cluster_centers_recursion(val.left, k, centroids, QoI_values)
                k = self._flatten_cluster_centers_recursion(val.right, k, centroids, QoI_values)
        return centroids, QoI_values


    def _flatten_cluster_centers_recursion(self, tree, k, centroids, QoI_values):
        if type(tree) == Cluster:
            centroids[k] = tree.params
            QoI_values[k,:] = tree.QoI_value
            k += 1
        elif type(tree) == ClusterTree:
            k = self._flatten_cluster_centers_recursion(tree.left, k, centroids, QoI_values)
            k = self._flatten_cluster_centers_recursion(tree.right, k, centroids, QoI_values)
        return k

    
    def predict(self, parameters: np.ndarray):
        predicted_vals = np.zeros(shape=(len(parameters),1+self.QoI_length))
        params_normalized = (parameters - self.mean.to_numpy()) / self.std.to_numpy()
        _, I = self.faiss_index.search(params_normalized, 1)
        I = np.matrix.flatten(I)
        predicted_vals[:,0] = I
        predicted_vals[:,1:] = self.QoI_values[I]
        return predicted_vals
    
    

    # Returns the QoI associated with the centroid of the cluster that param_row is located in
    # Finds which cluster we are in by using the tree structure from training the model
    def _search_tree(self, param_row: np.ndarray):
        dvec = np.zeros(len(self.tree))
        for j, base_element in np.ndenumerate(self.tree):
            dvec[j] = np.linalg.norm(base_element.params - param_row)
        k = np.argmin(dvec)
        return self._search_tree_recursion(param_row, self.tree[k])
        

    
    def _search_tree_recursion(self, param_row, tree):
        if type(tree) == Cluster:
            return tree.QoI_value, tree.index
        elif type(tree) == ClusterTree:
            left_distance = np.linalg.norm(tree.left.params - param_row)
            right_distance = np.linalg.norm(tree.right.params - param_row)
            if left_distance < right_distance:
                return self._search_tree_recursion(param_row, tree.left)
            else:
                return self._search_tree_recursion(param_row, tree.right)

