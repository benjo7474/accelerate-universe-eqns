from scipy.integrate import solve_ivp
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import torch
from RK2 import RK2

# These are our constants
N = 5  # Number of variables
F = 8  # Forcing

def L96(t,x):
    """Lorenz 96 model with constant forcing (from wikipedia)""" 
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

# Build L96 RHS in quadratic form (torch)
def build_Q():
    indices = np.zeros((2*N,3))
    values = np.zeros((2*N))
    for i in range(N):
        indices[2*i,:] = [(i+1)%N,(i-1)%N,i]
        indices[2*i+1,:] = [(i-2)%N,(i-1)%N,i]
        values[2*i] = 1.0
        values[2*i+1] = -1.0
    return torch.sparse_coo_tensor(indices=indices.transpose().tolist(), values=values.tolist(), dtype=torch.float32)

# L is just the negative identity
Q = build_Q()
def L96_2(t,x):
    return -x + F + torch.tensordot(Q.to_dense(), torch.outer(x,x), dims=([0,1],[0,1]))





### Solve ODE
x0 = F * torch.ones(N, dtype=torch.float32)  # Initial state (equilibrium)
x0[0] += 0.01  # Add small perturbation to the first variable

# x1 = solve_ivp(L96_2, (0.0,30.0), x0, method='RK45', rtol=1e-6)
x2 = solve_ivp(L96, (0.0,30.0), x0, method='RK45', rtol=1e-6)
tvec, xvec = RK2(L96_2, (0.0,30.0), 1e-3, x0)

# Plot the first three variables
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection="3d")
ax1.plot(xvec[0,:], xvec[1,:], xvec[2,:])
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$x_3$")
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.plot(x2.y[0,:], x2.y[1,:], x2.y[2,:])
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_zlabel("$x_3$")
plt.show()
