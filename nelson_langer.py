import numpy as np
from scipy.integrate import solve_ivp
from torch import sparse_coo_tensor
from torch.linalg import matmul

def build_tensors():

    # Parameters
    T: float = 1; # Temperature
    Av: float = 1; # Visual Extinction
    G0: float = 1; # Standard Interstellar Value
    shield: int = 1; # CO self-shielding factor
    n_h: float = 1e5; # hydrogen number density

    # s = [H_2, H_3^+, e, He, He^+, C, CH_x, O, OH_x, CO, HCO^+, C^+, M^+, M]
    #      0    1      2  3   4     5  6     7  8     9   10     11   12   13

    # Build 3rd order sparse tensor Q and 2nd order sparse matrix L
    indices_Q = [
        [0,1,2], #1
        [0,1,9], #2
        [0,1,5], #3
        [0,1,13], #4
        [0,1,7], #5
        [0,0,11], #6
        [0,0,4], #7
        [1,1,2], #8
        [1,1,9], #9
        [1,1,5], #10
        [1,1,13], #11
        [1,1,7], #12
        [2,10,2], #13
        [2,1,2], #14
        [2,12,2], #15
        [2,2,4], #16
        [2,2,11], #17
        [2,1,13], #18
        [3,2,4], #19
        [3,0,4], #20
        [3,9,4], #21
        [4,2,4], #22
        [4,0,4], #23
        [4,9,4], #24
        [5,2,11], #25
        [5,1,5], #26
        [5,8,5], #27
        [6,1,5], #28
        [6,6,7], #29
        [6,0,11], #30
        [7,1,7], #31
        [7,6,7], #32
        [7,9,4], #33
        [8,1,7], #34
        [8,8,11], #35
        [8,8,5], #36
        [9,10,2], #37
        [9,1,9], #38
        [9,6,7], #39
        [9,9,4], #40
        [9,8,5], #41
        [10,10,2], #42
        [10,1,9], #43
        [10,8,11], #44
        [11,2,11], #45
        [11,8,11], #46
        [11,0,11], #47
        [11,9,4], #48
        [12,12,2], #49
        [12,1,13], #50
        [13,12,2], #51
        [13,1,13] #52
    ]

    values_Q = [
        -1.9e-6/(T**0.54), #1
        1.7e-9, #2
        2e-9, #3
        2e-9, #4
        3e-10, #5
        -4e-16, #6
        -7e-15, #7
        -1.9e-6/(T**0.54), #8
        -1.7e-9, #9
        -2e-9, #10
        -2e-9, #11
        -8e-10, #12
        -3.3e-5/T, #13
        -1.9e-6/(T**0.54), #14
        -3.8e-10/(T**0.65), #15
        -9e-11/(T**0.64), #16
        -1.4e-10/(T**0.61), #17
        2e-9, #18
        9e-11/(T**0.64), #19
        7e-15, #20
        1.6e-9, #21
        -9e-11/(T**0.64), #22
        -7e-15, #23
        -1.6e-9, #24
        1.4e-10/(T**0.61), #25
        -2e-9, #26
        -5.8e-12*(T**0.5), #27
        2e-9, #28
        -2e-10, #29
        4e-16, #30
        -8e-10, #31
        -2e-10, #32
        1.6e-9, #33
        8e-10, #34
        -1e-9, #35
        -5.8e-12*(T**0.5), #36
        3.3e-5/T, #37
        -1.7e-9, #38
        2e-10, #39
        -1.6e-9, #40
        5.8e-12*(T**0.5), #41
        -3.3e-5/T, #42
        1.7e-9, #43
        1e-9, #44
        -1.4e-10/(T**0.61), #45
        -1e-9, #46
        -4e-16, #47
        1.6e-9, #48
        -3.8e-10/(T**0.65), #49
        2e-9, #50
        3.8e-10/(T**0.65), #51
        -2e-9 #52
    ]
    values_Q = values_Q * n_h

    indices_L = [
        [0,0],  #1
        [1,0],  #2
        [2,0],  #3
        [2,3],  #4
        [2,13], #5
        [2,5],  #6
        [3,3],  #7
        [4,3],  #8
        [5,6],  #9
        [5,5],  #10
        [5,9],  #11
        [6,6],  #12
        [7,8],  #13
        [7,9],  #14
        [8,8],  #15
        [9,10], #16
        [9,9],  #17
        [10,10],#18
        [11,5], #19
        [12,13],#20
        [13,13] #21
    ]

    values_L = [
        -1.2e-17, #1
        1.2e-17, #2
        1.2e-17, #3
        6.8e-18, #4
        2e-10*G0*np.exp(-1.9*Av), #5
        3e-10*G0*np.exp(-3*Av), #6
        -6.8e-18, #7
        6.8e-18, #8
        1e-9*G0*np.exp(-1.5*Av), #9
        -3e-10*G0*np.exp(-3*Av), #10
        1e-9*G0*shield*np.exp(-3*Av), #11
        -1e-9*G0*np.exp(-1.5*Av), #12
        5e-10*G0*np.exp(-1.7*Av), #13
        1e-9*G0*shield*np.exp(-3*Av), #14
        -5e-10*G0*np.exp(-1.7*Av), #15
        1.5e-10*G0*np.exp(-2.5*Av), #16
        -1e-9*G0*shield*np.exp(-3*Av), #17
        -1.5e-10*G0*np.exp(-2.5*Av), #18
        3e-10*G0*np.exp(-3*Av), #19
        2e-10*G0*np.exp(-1.9*Av), #20
        -2e-10*G0*np.exp(-1.9*Av) #21
    ]

    Q = sparse_coo_tensor(indices=indices_Q, values=values_Q, size=(14,14,14))
    L = sparse_coo_tensor(indices=indices_L, values=values_L, size=(14,14))

    return Q,L
        
def nelson_langer(t,x):
    N = 14
    Q,L = build_tensors()
    Q_dense = Q.to_dense()
    L_dense = L.to_dense()
    Q_reduced = np.zeros((N,N))
    for k in range(N):
        Q_reduced += x[k]*Q_dense[:,:,k]
    return matmul(L_dense + Q_reduced, x)

def jacobian(t,x):
    N = 14
    Q,L = build_tensors()
    for k in range(N):
        L += x[k]*(Q[:,:,k] + Q[:,k,:]).numpy()
    return L

x0 = 1e10 * np.ones(14)
soln = solve_ivp(nelson_langer, (0.0,1.0), x0, method='RK45')

# def Jacobian(t,s):
#     return np.array(tf.math.add(L*s, Q*np.outer(s,s)))

# tspan = [0, 1]
# s0 = [0,   0,     0, 0,  0,    0, 0,    0, 0,    0,  0,     0,   0,   0]
# #    [H_2, H_3^+, e, He, He^+, C, CH_x, O, OH_x, CO, HCO^+, C^+, M^+, M]

# sol = solve_ivp(Jacobian, t_span=tspan, y0=s0, method='RK45')