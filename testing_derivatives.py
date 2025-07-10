# %%
import numpy as np
from nelson_langer_network import build_nelson_network
import torch
import matplotlib.pyplot as plt
from test_clusters_and_plots import load_parameters


# %% check via finite differences if we get convergence
# only check individual entries since different scales messes with rounding errors
# major work in progress; lots of numerical instability

def build_Q(p):
    NL = build_nelson_network(params=p, compute_sensitivities=False)
    return NL.Q_dense
def build_L(p):
    NL = build_nelson_network(params=p, compute_sensitivities=False)
    return NL.L.to_dense()

def check_dQdp():
    p = torch.tensor(data=[10000, 80, 0.5])
    NL = build_nelson_network(params=p, compute_sensitivities=True)
    dQ_dp = NL.dQ_dp
    eye = torch.eye(3)
    hvec = 10.0**torch.arange(start=1,end=1,step=-1)
    errvec = torch.zeros(len(hvec))
    j = 0
    for i, h in enumerate(hvec):
        Q_pert_plus = build_Q(p + h*eye[j])
        Q_pert_minus = build_Q(p - h*eye[j])
        dQ_dp_fd = (Q_pert_plus[2,10,2] - Q_pert_minus[2,10,2]) / (2*h) #scalar
        errvec[i] = torch.abs(dQ_dp_fd - dQ_dp[2,10,2,j])

    import matplotlib.pyplot as plt
    plt.loglog(hvec, errvec)



# %% also check what the map p -> q looks like around a parameter p
# q in this case is xCO
def p_to_q_map(p: np.ndarray, tf):
    NL = build_nelson_network(params=p, compute_sensitivities=False)
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

    t, y = NL.solve_reaction([0, tf], x0, t_eval=[tf])
    return np.matrix.flatten(y)[9]

def p_to_dqdp_map(p: np.ndarray, tf):
    NL = build_nelson_network(params=p, compute_sensitivities=True)
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
    t, y = NL.solve_reaction([0, tf], x0, t_eval=[tf])
    return np.matrix.flatten(y)[[23, 37, 51]]



def p_to_dqdp_map_FD(p: np.ndarray, tf):

    pert = 1e-2 * p
    nh_pert = pert[0]
    T_pert = pert[1]
    G0_pert = pert[2]
    eye = np.eye(3)

    dq_dnh_fd = (p_to_q_map(p + nh_pert*eye[0], tf) - p_to_q_map(p - nh_pert*eye[0], tf)) / (2*nh_pert)
    dq_dT_fd = (p_to_q_map(p + T_pert*eye[1], tf) - p_to_q_map(p - T_pert*eye[1], tf)) / (2*T_pert)
    dq_dG0_fd = (p_to_q_map(p + G0_pert*eye[2], tf) - p_to_q_map(p - G0_pert*eye[2], tf)) / (2*G0_pert)
    dq_dp_fd = np.array([dq_dnh_fd, dq_dT_fd, dq_dG0_fd])
    return dq_dp_fd


def compute_error_from_FD(p, tf_years, print_vecs=False):
    # these parameters work pretty damn well
    # p = torch.tensor([10**0.062272, 10**4.017193, 0.741617])
    tf = 3600 * 24 * 365 * tf_years
    true = p_to_dqdp_map(p, tf)
    approx = p_to_dqdp_map_FD(p, tf)
    if print_vecs==True:
        print(f'True: {true}')
        print(f'FD approx: {approx}')
    return np.linalg.norm(true-approx)


def get_sample_of_parameters(N):
    # reads parameters, samples and converts to non-log format
    # returns in numpy format
    params = load_parameters()
    sampled_params = params.sample(N)
    sampled_params_numpy = sampled_params.to_numpy()
    sampled_params_numpy[:,[0,1]] = 10 ** sampled_params_numpy[:,[0,1]]
    return sampled_params_numpy



# %% Now include random sampling
# take 100 random parameters, run error for each and plot
def measure_FD_error_over_sample(N, tf_years):
    sample = get_sample_of_parameters(N)
    FD_errorvec = np.zeros(N)
    for j, param_row in enumerate(sample):
        FD_errorvec[j] = compute_error_from_FD(param_row, tf_years, print_vecs=False)

    plt.plot(FD_errorvec)
    return sample, FD_errorvec


def find_N_largest_indices(arr, N):

    # Get indices of the 10 largest entries
    indices = np.argpartition(arr, -N)[-N:]
    # Optional: sort these indices by actual values (descending)
    sorted_indices = indices[np.argsort(-arr[indices])]

    return sorted_indices


# %%
# x0 = np.array([0.5, 9.059e-9, 2e-4, 0.1, 7.866e-7, 0.0, 0.0, 0.0004, 0.0, 0.0, 0.0, 0.0002, 2.0e-7, 2.0e-7])
# secs_per_year = 3600*24*365
# years = 10000
# tf = years * secs_per_year

# NL = build_nelson_network(compute_sensitivities=True)
# t, y = NL.solve_reaction([0, tf], x0)

# %%
