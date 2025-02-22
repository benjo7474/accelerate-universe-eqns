# %%

import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from nelson_langer_network import nelson_langer_network


Q, L = nelson_langer_network.build_tensors()
Q_dense = Q.to_dense()
N = 14

def nelson_langer_wrapper(t, x: np.ndarray) -> np.ndarray:
    # takes a vector from numpy and returns vector from numpy
    xt = torch.from_numpy(x)
    return nelson_langer(xt).numpy()

def nelson_langer(x: torch.tensor) -> torch.tensor:
    # takes a vector from torch and does computation
    Q2 = torch.tensordot(Q_dense, x, dims=([2],[0]))
    return torch.linalg.matmul(Q2 + L, x)

def jacobian(x: torch.tensor) -> torch.tensor:
    # takes a vector from torch, returns torch
    return torch.tensordot(Q_dense, x, dims=([1],[0])) + torch.tensordot(Q_dense, x, dims=([2],[0])) + L

def jacobian_wrapper(t, x: np.ndarray) -> np.ndarray:
    # takes vector from numpy, returns numpy
    xt = torch.from_numpy(x)
    return jacobian(xt).numpy()

def check_normality(J: np.ndarray):
    # takes a rank 2 tensor in torch and returns ||J*(J^T) - (J^T)*J||
    Jtorch = torch.from_numpy(J)
    Jtranspose = torch.transpose(Jtorch, 0, 1)
    print(Jtranspose*J)
    return torch.linalg.matrix_norm(Jtorch*Jtranspose - Jtranspose*Jtorch, ord='fro')



# %% Run solution & plot

# Initial condition for NL99 from Despodic
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
tf = 30000 * secs_per_year
teval = np.linspace(start=0, stop=tf, num=5000)
soln = solve_ivp(nelson_langer_wrapper, [0,tf], x0, method='BDF',
            rtol=1e-16, jac=jacobian_wrapper, max_step=tf/100, t_eval=teval)

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

# Plot everything on log scale
fig, ax = plt.subplots()
for j in range(14):
    ax.plot(soln.t, soln.y[j,:], label=species[j])
ax.legend()
ax.set_yscale('log')
plt.show()

# NEW: take SVD of solution matrix for reduced order modeling purposes
U, S, Vh = np.linalg.svd(soln.y)


# %% Check normality of Jacobian matrix 
N = 100
norms = np.zeros(N)

for k in range(N):
    x = (n_h * np.random.rand(14) + 1) / n_h
    norms[k] = check_normality(jacobian_wrapper(0,x))

fig, ax = plt.subplots()
ax.plot(range(N), norms)
plt.show()
print(norms)



# %% Investigate eigenvalues of Jacobian
x = np.random.rand(14)
secs_per_year = 3600 * 24 * 365
soln = solve_ivp(nelson_langer_wrapper, [0, 3e4*secs_per_year], x0, method='RK45', rtol=1e-16)
J = jacobian_wrapper(0, soln.y[:,-1])
D, V = np.linalg.eig(J)

plt.semilogy(np.linspace(1, np.size(D), np.size(D)), np.absolute(D))
plt.show()
print(D)

# %%
