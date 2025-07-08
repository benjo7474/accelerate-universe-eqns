# Represents a cluster model for astro parameters.
# The class AstrochemClusterModel, after calling the function train_surrogate_model(),
# contains cluster centroids and QoI evaluations at those centroids that we can use
# k-nearest neighbors to obtain a good approximation.

from types import NoneType
import numpy as np
import pandas as pd
import sklearn.cluster
from nelson_langer_network import build_nelson_network
import faiss
from recursive_clustering import Cluster, ClusterTree, recursive_cluster_algorithm, flatten_cluster_centers
from taylor_grad_boost_KNN import GradientBoostModel


class AstrochemClusterModel:

    def __init__(self):
        self.N_clusters = 0
        self.model_trained = False



    def train_model(self,
        training_data: np.ndarray, # columns are log(n_h), log(T), G0 (not normalized)
        QoI: int,
        x0: np.ndarray,
        time: float,
        N: int = 10,
        ss: int = 10,
        do_clustering: bool = False,
        error_tol: float = 0.05,
        use_gradient_boost: bool = False,
        ode_solves: np.ndarray = None, 
        sensitivities: np.ndarray = None
    ):
        
        # if we need to do clustering, do this first.
        # this will come with ode solves
        # TODO: implement gradient?
        if do_clustering == True:
            k, cluster_tree = recursive_cluster_algorithm(
                training_data, lambda p: self._solve_nelson_network(p, x0, QoI, time),
                error_tol, N, ss, ode_solves
            )
            new_centroids, new_ode_solves = flatten_cluster_centers(k, 1, cluster_tree)
            self.KNN_model = GradientBoostModel(new_centroids, new_ode_solves)
            self.N_clusters = k

        # otherwise use every point in training_data as a centroid.
        else:

            # if ode_solves is None, we need to do the ode solves ourselves.
            # on top of this, if use_gradient_boost is true, we need to also save the gradients.
            # TODO: add multiple QoI
            if type(ode_solves) == NoneType:
                # do ode solves
                # save sensitivities as well if use_gradient_boost is true
                ode_solves = np.ndarray(len(training_data))
                if use_gradient_boost == True:
                    sensitivities = np.zeros(training_data.shape)
                else:
                    sensitivities = None
                for j, row in enumerate(training_data):
                    if use_gradient_boost == False:
                        ode_solves[j] = self._solve_nelson_network(row, x0, QoI, time)
                    elif use_gradient_boost == True:
                        ode_solves[j], sensitivities[j,:] = \
                            self._solve_nelson_network_with_sensitivities(row, x0, QoI, time)

                self.KNN_model = GradientBoostModel(training_data, ode_solves, sensitivities)
                self.N_clusters = len(training_data)
            
            else:
                # ode solves are saved in ode_solves
                # just create the faiss tree normally
                if use_gradient_boost == True:
                    self.KNN_model = GradientBoostModel(training_data, ode_solves, sensitivities)
                else:
                    self.KNN_model = GradientBoostModel(training_data, ode_solves)
                    
                self.N_clusters = len(training_data)
        
        self.model_trained = True



    # method takes in un-normalized parameters via numpy
    def _solve_nelson_network(self, params_row: np.ndarray, x0: np.ndarray, QoI: int, time: float):
        # undo the log
        n_h = 10 ** params_row[0]
        T = 10 ** params_row[1]
        G0 = params_row[2]
        # solve network
        network = build_nelson_network(params=np.array([n_h, T, G0]), compute_sensitivities=False)
        return network.solve_reaction_snapshot(x0, time, QoI)
        
    

    # also takes in un-normalized parameters via numpy
    def _solve_nelson_network_with_sensitivities(self, params_row: np.ndarray, x0: np.ndarray, QoI: int, time: float):
        n_h = 10 ** params_row[0]
        T = 10 ** params_row[1]
        G0 = params_row[2]
        network = build_nelson_network(params=np.array([n_h, T, G0]), compute_sensitivities=True)
        _, yvec = network.solve_reaction([0, time], x0, t_eval=[time])
        soln = yvec.flatten()
        grad_indices = np.array([14, 28, 42]) + QoI
        return soln[QoI], soln[grad_indices]

