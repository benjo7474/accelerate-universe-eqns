from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from torch import sparse_coo_tensor

# These are our constants
N = 5  # Number of variables
F = 8  # Forcing

def L96(t,x):
    """Lorenz 96 model with constant forcing (from wikipedia)""" 
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

# Build L96 RHS in quadratic form
def build_Q():
    indices = np.zeros((2*N,3))
    values = np.zeros((2*N))
    for i in range(N):
        indices[2*i,:] = [i,(i+1)%N,(i-1)%N]
        indices[2*i+1,:] = [i,(i-2)%N,(i-1)%N]
        values[2*i] = 1
        values[2*i+1] = -1
    return sparse_coo_tensor(indices=indices.transpose().tolist(), values=values.tolist())
# L is just the negative identity
def L96_2(t,x):
    sparse_Q = build_Q()
    Q = sparse_Q.to_dense()
    reduced_Q = np.zeros((N,N))
    for k in range(N):
        reduced_Q += x[k]*Q[:,:,k].numpy()
    return -x + F + np.matmul(reduced_Q,x)
def jacobian(t,x):
    sparse_Q = build_Q()
    Q = sparse_Q.to_dense()
    L = -1 * np.eye(N)
    for k in range(N):
        L += x[k]*(Q[:,:,k] + Q[:,k,:]).numpy()
    return L

x0 = F * np.ones(N)  # Initial state (equilibrium)
x0[0] += 0.01  # Add small perturbation to the first variable

x1 = solve_ivp(L96_2, (0.0,30.0), x0, method='RK45')
x2 = solve_ivp(L96_2, (0.0,30.0), x0, method='BDF', jac=jacobian)

# Plot the first three variables
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection="3d")
ax1.plot(x1.y[0,:], x1.y[1,:], x1.y[2,:])
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
