# %%
import numpy as np
from nelson_langer_network import build_nelson_network
from test_clusters_and_plots import load_parameters
import time


def get_N_parameters(N):
    all_params = load_parameters()
    sampled_params = all_params.sample(N)
    return sampled_params.to_numpy()


# this function takes in ONE row of parameters and spits out q and dqdp as np arrays, separately
def solve_for_sensitivities(p: np.ndarray, QoI: int, x0: np.ndarray, teval: np.ndarray):
    NL = build_nelson_network(p, compute_sensitivities=True)
    _, eval = NL.solve_reaction([0, teval[-1]], x0, t_eval=teval)
    sens_ind = 14 * np.arange(1,4) + QoI
    q = eval[QoI]
    dqdp = eval[sens_ind]
    return q, dqdp


if __name__ == '__main__':
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
    tf_years_eval = np.array([100, 1000, 10000, 40000])
    tf_eval = tf_years_eval * 365 * 24 * 3600
    
    parameters = np.load('TACC_code/input_p_small.npy')
    N = len(parameters)
    q_vec = np.empty((N, len(tf_eval)))
    dqdp_vec = np.empty(shape=(N, 3, len(tf_eval)))
    
    start = time.perf_counter()
    for j, p in enumerate(parameters):
        q, dqdp = solve_for_sensitivities(p, 9, x0, tf_eval) # solve for xCO
        q_vec[j] = q
        dqdp_vec[j] = dqdp

    end = time.perf_counter()

    print(f'Time elapsed: {end-start:.2f} seconds.')

    # np.save('input_p.npy', parameters)
    np.save('output_q_CO_test.npy', q_vec)
    np.save('output_dqdp_CO_test.npy', dqdp_vec)

    

# %%
