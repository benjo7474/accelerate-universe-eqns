#### This is the code that I am using for essentially everything
#### At some point need to turn this into non-spaghetti code
# Or just say this has the form of a notebook. Maybe I should turn this into a jupyter notebook...


# %%
import numpy as np
import pandas as pd
import time
from astrochem_clustering import AstrochemClusterModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

QoI = 9
ode_solve_columns = [8, 9, 10, 11, 12]


# ---- FUNCTIONS TO LOAD DATA ---- #


# Load medium dataset (with ode solves)
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



# Load large dataset (with ode solves)
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


# Load FULL dataset (with ODE solves!)
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




# ---- FUNCTIONS TO BUILD CLUSTER MODELS ---- #


def test_model(N_train=10000, N_test=5000, tf_index=2, k=1):

    # load data
    p = np.load('TACC_code/input_p_full.npy') # this is in normal form
    p[:,[0,1]] = np.log10(p[:,[0,1]])
    q = np.load('TACC_code/output_q_CO_full.npy')
    dqdp = np.load('TACC_code/output_dqdp_CO_full.npy')

    # 0.1K years = 0
    # 1K years = 1
    # 10K years = 2
    # 40K years = 3
    q_tf = q[:,tf_index]
    dqdp_tf = dqdp[:,:,tf_index]

    # pick points to use in data
    indices = np.arange(len(p), dtype=int)
    rand_inds = np.random.choice(indices, N_train + N_test, replace=False)
    train_inds = rand_inds[:N_train]
    test_inds = rand_inds[N_train:]

    surrogate = AstrochemClusterModel()
    surrogate.train_model(p[train_inds], 9, x0, tf,
                          use_gradient_boost=True,
                          do_clustering=False,
                          ode_solves=q_tf[train_inds],
                          sensitivities=dqdp_tf[train_inds])

    percent_error, absolute_error, _, _ = surrogate.test_accuracy(p[test_inds], q_tf[test_inds], k)
    bad_p_ind = np.argmax(percent_error)
    bad_p = p[test_inds][bad_p_ind]
    # good_p_ind = np.argmin(percent_error)
    # good_p = p[test_inds][good_p_ind]
    surrogate.convex_NN_plot(bad_p)
    surrogate.generate_slice_plots(bad_p)

    surrogate.plot_point_cloud(p[train_inds], p[test_inds])
    



def train_and_test_model(tf_index=2):
    
    p = np.load('TACC_code/input_p_medium.npy') # not log
    p[:,[0,1]] = np.log10(p[:,[0,1]])
    q = np.load('TACC_code/output_q_CO_medium.npy')
    dqdp = np.load('TACC_code/output_dqdp_CO_medium.npy')

    # randomize order
    np.random.seed(1234)
    rand_inds = np.random.choice(np.arange(len(p)), len(p), replace=False)
    p = p[rand_inds]
    q = q[rand_inds]
    dqdp = dqdp[rand_inds]

    # uniformly sampled points
    N_train = 8000
    N_test = len(p) - N_train
    
    p_train = p[:N_train]
    q_train = q[:N_train,tf_index]
    dqdp_train = dqdp[:N_train,:,tf_index]
    p_test = p[N_train:(N_train+N_test)]
    q_test = q[N_train:(N_train+N_test),tf_index]

    surrogate = AstrochemClusterModel()
    start = time.perf_counter()

    # Do clustering from scratch
    # surrogate.train_model(p_train, 9, x0, tf, error_tol=0.01,
    #                       do_clustering=True, use_gradient_boost=True, N=50,
    #                       ode_solves=q_train, sensitivities=dqdp_train)
    
    # Already clustered
    features = np.load('input_p_clustered_medium.npy')
    labels = np.load('output_q_clustered_medium.npy')
    gradients = np.load('output_dqdp_clustered_medium.npy')
    surrogate.train_model(features, 9, x0, tf, do_clustering=False, use_gradient_boost=True,
                          ode_solves=labels, sensitivities=gradients)
    
    # Do no clustering
    # surrogate2 = AstrochemClusterModel()
    # surrogate2.train_model(p_train, 9, x0, tf, do_clustering=False, use_gradient_boost=True,
    #                        ode_solves=q_train, sensitivities=dqdp_train)

    stop = time.perf_counter()
    print(f'Time to cluster: {stop-start:.2f} seconds')

    rel_err, _, _, _ = surrogate.test_accuracy(p_test, q_test, 1)
    len(p_test)
    len(rel_err)
    # good_p_ind = np.argmin(rel_err)
    # good_p = p_test[good_p_ind]
    order = np.argsort(rel_err)
    print(p_test[order])
    bad_p_ind = order[-1]
    bad_p = p_test[bad_p_ind]
    surrogate.convex_NN_plot(bad_p)
    surrogate.generate_slice_plots(bad_p)

    surrogate.plot_point_cloud(p_train)

    # save the new source points
    # np.save('input_p_clustered_medium.npy', surrogate.KNN_model.features)
    # np.save('output_q_clustered_medium.npy', surrogate.KNN_model.labels)
    # np.save('output_dqdp_clustered_medium.npy', surrogate.KNN_model.gradients)




# K times: pick 80% of dataset as training and 20% as testing
# for each set of training data, run each tol, for both grad and w/o grad
# save N_clusters, error statistics (max, mean, median, average, histogram) in table
# put histograms in PDF with well-labeled titles and axes
def convergence_study(
    input_name = 'data/input_p_full.npy',
    output_name = 'data/output_q_CO_full.npy',
    grads_name = 'data/output_dqdp_CO_full.npy',
    N_points = 1000000,
    rel_errs = [0.2, 0.1, 0.05, 0.01, 0.001],
    K = 5,
    N_init = 10,
    pdf_name = 'convergence_study_plots.pdf'
):
    
    # load data
    p = np.load(input_name) # not log
    p[:,[0,1]] = np.log10(p[:,[0,1]])
    q = np.load(output_name)
    dqdp = np.load(grads_name)

    # filter
    randinds = np.random.choice(np.arange(len(p)), N_points, replace=False)
    p = p[randinds]
    q = q[randinds]
    dqdp = dqdp[randinds]

    # pick only one time horizon (10K years for now)
    tf_index = 2
    q_tf = q[:,tf_index]
    dqdp_tf = dqdp[:,:,tf_index]


    with PdfPages(pdf_name) as pdf:

        # loop over K times and pick training and testing data split
        # we do this because we want to use the same split for different tolerances
        for i in range(K):

            N_train = 8 * len(p) // 10   # 80 pct of data
            sample_inds = np.arange(len(p))
            np.random.shuffle(sample_inds)

            p_train = p[sample_inds[:N_train]]
            q_train = q_tf[sample_inds[:N_train]]
            dqdp_train = dqdp_tf[sample_inds[:N_train]]
            p_test = p[sample_inds[N_train:]]
            q_test = q_tf[sample_inds[N_train:]]
            # dqdp_test = dqdp_tf[sample_inds[N_train:]]

            # for each tolerance,
            for tol in rel_errs:
                
                start_time = time.perf_counter()

                # first for non-gradient model
                if tol >= 0.01: # since it is very slow if we do not do this
                    no_gradient_model = AstrochemClusterModel()
                    no_gradient_model.train_model(
                        p_train, QoI, x0, tf,
                        N = N_init,
                        do_clustering = True,
                        error_tol = tol,
                        use_gradient_boost = False,
                        ode_solves = q_train,
                    )
                    N_ng = no_gradient_model.N_clusters

                    _, _, fig_g_rel, fig_g_abs = no_gradient_model.test_accuracy(p_test, q_test, 1,
                            f'Relative Errors (No Gradient), $tol={tol}$, $N={N_ng}$',
                            print_stats=False, disp_figs=False)
                    # save to pdf
                    pdf.savefig(fig_g_rel)
                    plt.close(fig_g_rel)
                    plt.close(fig_g_abs)

                    mid_time = time.perf_counter()
                    print(f'Time for no gradient, tol={tol}, attempt {i}: {mid_time-start_time} seconds')

                # then for gradient model
                gradient_model = AstrochemClusterModel()
                gradient_model.train_model(
                    p_train, QoI, x0, tf,
                    N = N_init,
                    do_clustering = True,
                    error_tol = tol,
                    use_gradient_boost = True,
                    ode_solves = q_train,
                    sensitivities = dqdp_train
                )
                N_g = gradient_model.N_clusters

                # statistics
                _, _, fig_ng_rel, fig_ng_abs = gradient_model.test_accuracy(p_test, q_test, 1,
                        f'Relative Errors (With Gradient), $tol={tol}$, $N={N_g}$',
                        print_stats=False, disp_figs=False)
                
                # save to pdf
                pdf.savefig(fig_ng_rel)
                plt.close(fig_ng_rel)
                plt.close(fig_ng_abs)

                end_time = time.perf_counter()
                if tol >= 0.01:
                    print(f'Time for with gradient, tol={tol}, attempt {i}: {end_time-mid_time} seconds')
                else:
                    print(f'Time for with gradient, tol={tol}, attempt {i}: {end_time-start_time} seconds')




def compare_uniform_vs_adaptive_sampling(tf_index=2):

    # load data
    p = np.load('data/input_p.npy')
    p[:,[0,1]] = np.log10(p[:,[0,1]])
    q = np.load('data/output_q_CO.npy')
    dqdp = np.load('data/output_dqdp_CO.npy')

    q_tf = q[:,tf_index]
    dqdp_tf = dqdp[:,:,tf_index]

    # put aside some of the data for testing
    N_test = 10000
    p_test = p[:N_test]
    q_test = q_tf[:N_test]
    p_use = p[N_test:]
    q_use = q_tf[N_test:]
    dqdp_use = dqdp_tf[N_test:]

    # number of samples we compute uniformly
    N_uniform = 30000
    indices = np.arange(len(p_use), dtype=int)
    np.random.seed(1234)
    rand_inds = np.random.choice(indices, N_uniform, replace=False)
    p_uniform_train = p_use[rand_inds]
    q_uniform_train = q_use[rand_inds]
    dqdp_uniform_train = dqdp_use[rand_inds]
    
    # "train" uniform model
    surrogate_uniform = AstrochemClusterModel()
    surrogate_uniform.train_model(p_uniform_train, 9, x0, tf,
                                  do_clustering=False,
                                  use_gradient_boost=True,
                                  ode_solves=q_uniform_train,
                                  sensitivities=dqdp_uniform_train)
    
    # train adaptive model
    # start = time.perf_counter()
    surrogate_adaptive = AstrochemClusterModel()
    # surrogate_adaptive.train_model(p_use, 9, x0, tf,
    #                                do_clustering=True,
    #                                N=40,
    #                                use_gradient_boost=True,
    #                                error_tol=0.01,
    #                                ode_solves=q_use,
    #                                sensitivities=dqdp_use)
    # stop = time.perf_counter()
    features = np.load('features_full.npy')
    labels = np.load('labels_full.npy')
    gradients = np.load('gradients_full.npy')
    surrogate_adaptive.train_model(features, 9, x0, tf,
                                   do_clustering=False,
                                   use_gradient_boost=True,
                                   ode_solves=labels,
                                   sensitivities=gradients)
    # print(f'Time to run adaptive clustering algorithm: {stop-start:.2f} seconds')
    
    # save points in adaptive algorithm
    # np.save('features_full.npy', surrogate_adaptive.KNN_model.features)
    # np.save('labels_full.npy', surrogate_adaptive.KNN_model.labels)
    # np.save('gradients_full.npy', surrogate_adaptive.KNN_model.gradients)

    # test algorithms
    rel_err_uniform, _ = surrogate_uniform.test_accuracy(p_test, q_test, title='Rel. Err. with Uniform Sampling')
    rel_err_adaptive, _ = surrogate_adaptive.test_accuracy(p_test, q_test, title='Rel. Err. with Adaptive Sampling')

    # plot point cloud
    surrogate_adaptive.plot_point_cloud(p_uniform_train, q_uniform_train)

    # play with large errors
    order = np.argsort(rel_err_adaptive)
    print(p_test[order])
    bad_p_ind = order[-1]
    bad_p = p_test[bad_p_ind]
    surrogate_adaptive.convex_NN_plot(bad_p)
    surrogate_adaptive.generate_slice_plots(bad_p)


def check_ode_solves_from_TACC(N=20, QoI=9):

    p = np.load('TACC_code/input_p_medium.npy')
    q = np.load('TACC_code/output_q_CO_medium.npy')
    dqdp = np.load('TACC_code/output_dqdp_CO_medium.npy')
    print(p.shape)
    print(q.shape)
    print(dqdp.shape)
    i = np.random.randint(len(p), size=N)
    # i = np.arange(N)
    from generate_nelson_data import solve_for_sensitivities
    tf_years_eval = np.array([100, 1000, 10000, 40000])
    tf_eval = 3600 * 365 * 24 * tf_years_eval

    for ind in i:
        p1 = p[ind]
        q1 = q[ind]
        dqdp1 = dqdp[ind]
        q2, dqdp2 = solve_for_sensitivities(p1, QoI, x0, tf_eval)

        print(f'p={p1}')
        print(q1)
        print(q2)
        print(dqdp1)
        print(dqdp2)

    print(np.abs(q1-q2).max())
    print(np.abs(dqdp1-dqdp2).max())
    


# If we want to save the surrogate, we need to use pickle 
def save_surrogate(surrogate, filename):
    import pickle
    with open(filename, 'wb') as file:
        pickle.dump(surrogate, file)

# Load the tree back in with pickle
def load_surrogate(filename):
    import pickle
    with open(filename, 'rb') as file:
        surrogate = pickle.load(file)
        return surrogate


# Test Model
def test_from_surrogate_object_no_ode_solves(surrogate, test):

    datamat = np.zeros(shape=(len(test),24))
    # Exact solution for e, H2, H3+, CO, C in columns 0-4
    # Index in column 5
    # Predictions in columns 6-10
    # Relative errors in columns 11-15
    # Absolute errors in columns 16-20
    # Test data in columns 21-23

    # exact solution
    datamat[:,[0,1,2,3,4]] = test.to_numpy()[:, [8,9,10,11,12]]
    # index and predictions
    datamat[:,[5,6,7,8,9,10]] = surrogate.predict(test[['$\log(n_h)$','$\log(T)$','$G_0$']].to_numpy())
    # percent errors
    datamat[:,[11,12,13,14,15]] = np.abs(datamat[:,[6,7,8,9,10]] - datamat[:,[0,1,2,3,4]]) / np.abs(datamat[:,[0,1,2,3,4]])
    # relative errors
    datamat[:,[16,17,18,19,20]] = np.abs(datamat[:,[6,7,8,9,10]] - datamat[:,[0,1,2,3,4]])
    # test data
    datamat[:,[21,22,23]] = test[['$\log(n_h)$','$\log(T)$','$G_0$']].to_numpy()

    return datamatrix_to_pandas(datamat)



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


if __name__ == '__main__':
    # test_model(N_train=200000, N_test=10000, k=1, tf_index=2)
    # compare_uniform_vs_adaptive_sampling()
    # train_and_test_model()
    convergence_study()


# # %%
# train, test = load_full_dataset_with_ode_solves()
# test = test.sample(100).to_numpy()

# # %%
# surrogate = train_model()
# surrogate.KNN_model.test_accuracy(test[:,0:3], test.to_numpy()[:,16], 1)
# # %%
# test_taylor_grad_boost_KNN_class()



# %%
######## ------ PLOTS AND ERROR ANALYSIS ------ ########



# Error statistics
def print_error_statistics(datamat: pd.DataFrame):
    ind = 11
    print('Percent error for e')
    # print(f'Mean: {np.mean(datamat['Percent Error e'])}')
    print(f'Median: {np.median(datamat[:,ind])}')
    print(f'Max: {np.max(datamat[:,ind])}')
    print(f'STD: {np.std(datamat[:,ind])}')
    tol = 0.1
    num_vals_outside_error = (datamat[:,ind] > tol).sum()
    print(f'# of points outside {tol*100}% error: {num_vals_outside_error} ({num_vals_outside_error/len(datamat[:,2])}% of data)\n')

    ind = 12
    print('Stats for H2')
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

