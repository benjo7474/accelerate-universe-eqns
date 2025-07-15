# Represents a cluster model for astro parameters.
# The class AstrochemClusterModel, after calling the function train_surrogate_model(),
# contains cluster centroids and QoI evaluations at those centroids that we can use
# k-nearest neighbors to obtain a good approximation.

from types import NoneType
import numpy as np
from nelson_langer_network import build_nelson_network
from recursive_clustering import \
    recursive_cluster_algorithm, flatten_cluster_centers, \
    flatten_cluster_centers_with_gradient
from taylor_grad_boost_KNN import GradientBoostModel
import matplotlib.pyplot as plt


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
        
        self.x0 = x0
        self.QoI = QoI
        self.time = time
        self.p_to_q = lambda p: self._solve_nelson_network(p, x0, QoI, time)
        self.p_to_q_and_dqdp = lambda p: self._solve_nelson_network_with_sensitivities(p, x0, QoI, time)
        
        # if we need to do clustering, do this first.
        # this will come with ode solves
        if do_clustering == True:
            if use_gradient_boost == False:
                k, cluster_tree = recursive_cluster_algorithm(
                    training_data, self.p_to_q, error_tol, N, ss, use_gradient_boost,
                    [True, True, False], ode_solves, sensitivities)
                new_centroids, new_ode_solves = flatten_cluster_centers(k, cluster_tree)
                self.KNN_model = GradientBoostModel(new_centroids, new_ode_solves)
            elif use_gradient_boost == True:
                k, cluster_tree = recursive_cluster_algorithm(
                    training_data, self.p_to_q_and_dqdp, error_tol, N, ss, use_gradient_boost,
                    [True, True, False], ode_solves, sensitivities)
                new_centroids, new_ode_solves, new_sensitivities = flatten_cluster_centers_with_gradient(k, cluster_tree)
                self.KNN_model = GradientBoostModel(new_centroids, new_ode_solves, new_sensitivities)
            
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
                        ode_solves[j] = self.p_to_q(row)
                    elif use_gradient_boost == True:
                        ode_solves[j], sensitivities[j,:] = self.p_to_q_and_dqdp(row)

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


    def predict(self, targets: np.ndarray, k=1, disp_time=False):
        if targets.ndim == 1:
            return self.KNN_model.predict(np.reshape(targets, (1,3)), k, unwrap_log=[True, True, False], disp_time=True)
        else:
            return self.KNN_model.predict(targets, k, unwrap_log=[True, True, False], disp_time=True)


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


    ### --- TESTS --- ###

    # given a matrix of features and a list of true values, use above function to predict,
    # followed by comparing the accuracy to true_values and displaying error statistics.
    def test_accuracy(self, targets: np.ndarray, true_values: np.ndarray, k: int = 1, title='Relative Error'):

        # TODO check if length of targets and true_values matches
        # TODO check if length of each feature vector matches length of training vectors

        predicted_values = self.predict(targets, disp_time=True)
        absolute_error = np.abs(predicted_values - true_values)
        percent_error = absolute_error / np.abs(true_values)

        print('PERCENT ERRORS')
        print(f'Mean: {np.mean(percent_error)}')
        print(f'Median: {np.median(percent_error)}')
        print(f'Max: {np.max(percent_error)}')
        print(f'STD: {np.std(percent_error)}')
        tol = 0.01
        num_vals_outside_error = (percent_error > tol).sum()
        print(f'# of points outside {tol*100}% error: {num_vals_outside_error} ({num_vals_outside_error/len(percent_error)}% of data)\n')

        plt.figure()
        plt.hist(percent_error, bins=np.arange(0, 1.01*percent_error.max(), 0.01*percent_error.max()))
        plt.yscale('log')
        plt.title(f'{title} ($N_s = {self.N_clusters}, N_t = {len(targets)}$)', fontsize=14)
        plt.xlabel('Relative Error', fontsize=12)
        plt.ylabel('# Points', fontsize=12)
        plt.text(1.01*percent_error.max(), 2e3,
                f'Mean: {np.mean(percent_error):.3e}\n\
                Median: {np.median(percent_error):.3e}\n\
                Max: {np.max(percent_error):.3e}\n\
                STD: {np.std(percent_error):.3e}',
                fontsize=10, horizontalalignment='right')
        plt.show()

        print('ABSOLUTE ERRORS')
        print(f'Mean: {np.mean(absolute_error)}')
        print(f'Median: {np.median(absolute_error)}')
        print(f'Max: {np.max(absolute_error)}')
        print(f'STD: {np.std(absolute_error)}\n')

        plt.figure()
        plt.hist(np.log10(absolute_error))
        plt.yscale('log')
        plt.title('Absolute Error xCO (with gradient)')
        plt.show()

        return percent_error, absolute_error

    
    # plot the convex combination from the closest source to the target
    # also plot the slices (1% in each direction)
    def convex_NN_plot(self, p: np.ndarray):

        # compute nearest neighbor and compute directional derivative from this point
        D, I = self.KNN_model._faiss_index.search(np.reshape(p, (1,3)), 1)
        closest_point = self.KNN_model.features[I[0,0]]
        closest_value = self.KNN_model.labels[I[0,0]]
        print(f'Point: {p}')
        prediction = self.predict(np.reshape(p, (1,3)), 1)

        print(f'Point: {p}')
        print(f'Nearest neighbor: {closest_point}')

        alphavec = np.linspace(-0.05, 1.05, 200)
        convex_grid = np.outer((1-alphavec), closest_point) + np.outer(alphavec, p)
        qvec = np.zeros(len(convex_grid))
        for i, val in enumerate(convex_grid):
            qvec[i] = self.p_to_q(val)
        
        plt.figure()
        plt.plot(alphavec, qvec, label='$q$')
        plt.xlabel('Nearest neighbor to parameter of interest')
        plt.ylabel('$q$')
        plt.title(f'Plot from NN {np.array2string(closest_point, precision=2)} to p {np.array2string(p, precision=2)}')
        plt.xlim([-0.05, 1.05])
        plt.scatter([1], [closest_value], marker='x', label='0th order approx.')
        plt.scatter([1], [prediction], marker='x', label='1st order approx. (grad boost)')
        plt.scatter([1], [self.p_to_q(p)], marker='x', label='Exact')
        plt.legend()
        plt.show()

    
    def generate_slice_plots(self, p):

        pert = 0.1 * p
        nh_pert = pert[0]
        T_pert = pert[1]
        G0_pert = pert[2]
        eye = np.eye(3)

        # slice in n_h
        nh_pert_vec = np.linspace(start=-nh_pert, stop=nh_pert, num=100)
        qvec1 = np.zeros(len(nh_pert_vec))
        for i, pert in enumerate(nh_pert_vec):
            qvec1[i] = self.p_to_q(p + pert*eye[0])

        # slice in T
        T_pert_vec = np.linspace(start= -T_pert, stop=T_pert, num=100)
        qvec2 = np.zeros(len(T_pert_vec))
        for i, pert in enumerate(T_pert_vec):
            qvec2[i] = self.p_to_q(p + pert*eye[1])

        # slice in G0
        G0_pert_vec = np.linspace(start= -G0_pert, stop=G0_pert, num=100)
        qvec3 = np.zeros(len(G0_pert_vec))
        for i, pert in enumerate(G0_pert_vec):
            qvec3[i] = self.p_to_q(p + pert*eye[2])

        # plot
        plt.figure()
        plt.plot(p[0] + nh_pert_vec, qvec1)
        plt.ylim(qvec1.min(), qvec1.max())
        plt.xlabel('$\log(n_h)$')
        plt.ylabel('$q$')
        plt.title('$\\log(n_h)$ Slice Plot ($\\log(T)$ and $G_0$ fixed)')
        plt.show()

        plt.figure()
        plt.plot(p[1] + T_pert_vec, qvec2)
        plt.ylim(qvec2.min(), qvec2.max())
        plt.xlabel('$\log(T)$')
        plt.ylabel('$q$')
        plt.title('$\\log(T)$ Slice Plot ($\\log(n_h)$ and $G_0$ fixed)')
        plt.show()

        plt.figure()
        plt.plot(p[2] + G0_pert_vec, qvec3, label='$\\log(n_h)$ and $\\log(T)$ fixed')
        plt.ylim(qvec2.min(), qvec2.max())
        plt.xlabel('$G_0$')
        plt.ylabel('$q$')
        plt.title('$G_0$ Slice Plot ($\\log(T)$ and $\\log(n_h)$ fixed)')
        plt.show()


    
    def plot_point_cloud(self, train_points, test_points):

        if self.model_trained == True:

            plt.figure()
            plt.scatter(train_points[:,0], train_points[:,1],
                        label=f'Training points using uniform sampling ($N={len(train_points)}$)', marker='o', alpha=0.3, c='red', s=20)
            # plt.scatter(test_points[:,0], test_points[:,1],
            #             label=f'Testing Data ($N={len(test_points)}$)', marker='o', alpha=0.3, c='green', s=20)
            plt.scatter(self.KNN_model.features[:,0], self.KNN_model.features[:,1],
                        label=f'Training points using adaptive clustering ($N={len(self.KNN_model.features)}$)', marker='.', c='blue', s=15)
            plt.legend()
            plt.xlabel('$\\log_{10}(n_h)$', fontsize=12)
            plt.ylabel('$\\log_{10}(T)$', fontsize=12)
            plt.title('Uniform vs. Adaptive Sampling', fontsize=13)
            plt.show()


