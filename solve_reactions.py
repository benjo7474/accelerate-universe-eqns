# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from nelson_langer_network import build_nelson_network


def check_normality(J: np.ndarray):
    # takes a rank 2 tensor in torch and returns ||J*(J^T) - (J^T)*J||
    Jtorch = torch.from_numpy(J)
    Jtranspose = torch.transpose(Jtorch, 0, 1)
    print(Jtranspose*J)
    return torch.linalg.matrix_norm(Jtorch*Jtranspose - Jtranspose*Jtorch, ord='fro')



# %% Run solution & plot

# Model parameters
Av: float = 2; # Visual Extinction
G0: float = 1.7; # Standard Interstellar Value
shield: int = 1; # CO self-shielding factor
n_h: float = 611; # hydrogen number density
T: float = 10; # Temperature

# Can also load random params for n_h and T
A = np.load('np-states.npy')
j = np.random.randint(low=0, high=A.shape[0])
n_h = 2*A[j,0]
T = A[j,1]
G0 = A[j,3]  # /8.94e-14 ?
print(f'n_h: {n_h} \t T: {T} \t G0: {G0}')

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
# teval = np.linspace(start=0, stop=tf, num=5000)
nelson_langer_network = build_nelson_network()
soln = nelson_langer_network.solve_reaction([0, tf], x0)

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


# %% Check conservation conditions

# charge conservation: n(e) = n(H_3^+) + n(He^+) + n(HCO^+) + n(C^+) + n(M^+)
net_charge = soln.y[1,:] + soln.y[4,:] + soln.y[10,:] + soln.y[11,:] + soln.y[12,:] - soln.y[2,:]
# helium conservation: n(He) + n(He^+) = constant
He_total = soln.y[3,:] + soln.y[4,:]
# metal conservation: n(M) + n(M^+) = constant
M_total = soln.y[12,:] + soln.y[13,:]
# carbon conservation: n(C) + n(C^+) + n(CH_x) + n(C0) + n(HCO^+)
C_total = soln.y[5,:] + soln.y[6,:] + soln.y[9,:] + soln.y[10,:] + soln.y[11,:]
# oxygen conservation: n(O) + 
O_total = soln.y[7,:] + soln.y[8,:] + soln.y[9,:] + soln.y[10,:]
# hydrogen conservation: skeptical this works because we don't track H in the reactions
H_total = soln.y[0,:] + soln.y[1,:] + soln.y[6,:] + soln.y[8,:] + soln.y[10,:]

# Can (and will) make these equations a matrix but it is kind of a lot of work for now

# %% Check normality of Jacobian matrix 
N = 100
norms = np.zeros(N)

for k in range(N):
    x = np.random.rand(14)
    norms[k] = check_normality(nelson_langer_network.jacobian(0,x))

fig, ax = plt.subplots()
ax.plot(range(N), norms)
plt.show()
print(norms)



# %% Investigate eigenvalues of Jacobian
x = np.random.rand(14)
secs_per_year = 3600 * 24 * 365
soln = nelson_langer_network.solve_reaction([0, 3e4*secs_per_year], x0)
J = nelson_langer_network.jacobian(0, soln.y[:,-1])
D, V = np.linalg.eig(J)

plt.semilogy(np.linspace(1, np.size(D), np.size(D)), np.absolute(D))
plt.show()
print(D)

# %%
