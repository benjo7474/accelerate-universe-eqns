from reactions import Reaction, ReactionNetwork
import numpy as np
import torch


Av: float = 2; # Visual Extinction
G0: float = 1.7; # Standard Interstellar Value
shield: int = 1; # CO self-shielding factor
n_h: float = 611; # hydrogen number density
T: float = 10; # Temperature

# Can also load random params for n_h and T
A = np.load('data/tracer_parameter_data.npy')
j = np.random.randint(low=0, high=A.shape[0])
# n_h = 2*A[j,0]
# T = A[j,1]
# print(f'n_h: {n_h} \t T: {T}')

def build_tensors():

    # [H_2, H_3^+, e, He, He^+, C, CH_x, O, OH_x, CO, HCO^+, C^+, M^+, M]
    #  0    1      2  3   4     5  6     7  8     9   10     11   12   13

    indices_Q = np.array([
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
    ])

    values_Q = np.array([
        1.9e-6/(T**0.54), #1 g
        1.7e-9, #2 g
        2e-9, #3 g
        2e-9, #4 g
        8e-10, #5 g
        -4e-16, #6 g
        -7e-15, #7 g
        -1.9e-6/(T**0.54), #8 g
        -1.7e-9, #9 g
        -2e-9, #10 g
        -2e-9, #11 g
        -8e-10, #12 g
        -3.3e-5/T, #13 g
        -1.9e-6/(T**0.54), #14 g
        -3.8e-10/(T**0.65), #15 g
        -9e-11/(T**0.64), #16 g
        -1.4e-10/(T**0.61), #17 g
        2e-9, #18 g
        9e-11/(T**0.64), #19 g
        7e-15, #20 g
        1.6e-9, #21 g
        -9e-11/(T**0.64), #22 g
        -7e-15, #23 g
        -1.6e-9, #24 g
        1.4e-10/(T**0.61), #25 g
        -2e-9, #26 g
        -5.8e-12*(T**0.5), #27 g
        2e-9, #28 g
        -2e-10, #29 g
        4e-16, #30 g
        -8e-10, #31 g
        -2e-10, #32 g
        1.6e-9, #33 g
        8e-10, #34 g
        -1e-9, #35 g
        -5.8e-12*(T**0.5), #36 g
        3.3e-5/T, #37 g
        -1.7e-9, #38 g
        2e-10, #39 g
        -1.6e-9, #40 g
        5.8e-12*(T**0.5), #41 g
        -3.3e-5/T, #42 g
        1.7e-9, #43 g
        1e-9, #44 g
        -1.4e-10/(T**0.61), #45 g
        -1e-9, #46 g
        -4e-16, #47 g
        1.6e-9, #48 g
        -3.8e-10/(T**0.65), #49 g
        2e-9, #50 g
        3.8e-10/(T**0.65), #51 g
        -2e-9 #52 g
    ])
    values_Q = values_Q * n_h

    indices_L = np.array([
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
    ])

    values_L = np.array([
        -1.2e-17, #1 g
        1.2e-17, #2 g
        1.2e-17, #3 g
        6.8e-18, #4 g
        2e-10*G0*np.exp(-1.9*Av), #5 g
        3e-10*G0*np.exp(-3*Av), #6 g
        -6.8e-18, #7 g
        6.8e-18, #8 g
        1e-9*G0*np.exp(-1.5*Av), #9 g
        -3e-10*G0*np.exp(-3*Av), #10 g
        1e-10*G0*shield*np.exp(-3*Av), #11 g
        -1e-9*G0*np.exp(-1.5*Av), #12 g
        5e-10*G0*np.exp(-1.7*Av), #13 g
        1e-10*G0*shield*np.exp(-3*Av), #14 g
        -5e-10*G0*np.exp(-1.7*Av), #15 g
        1.5e-10*G0*np.exp(-2.5*Av), #16 g
        -1e-10*G0*shield*np.exp(-3*Av), #17 g
        -1.5e-10*G0*np.exp(-2.5*Av), #18 g
        3e-10*G0*np.exp(-3*Av), #19 g
        2e-10*G0*np.exp(-1.9*Av), #20 g
        -2e-10*G0*np.exp(-1.9*Av) #21 g
    ])

    Q = torch.sparse_coo_tensor(indices=indices_Q.transpose().tolist(), values=values_Q.transpose().tolist(), size=(14,14,14), dtype=torch.float64)
    L = torch.sparse_coo_tensor(indices=indices_L.transpose().tolist(), values=values_L.transpose().tolist(), size=(14,14), dtype=torch.float64)

    return Q,L



def build_nelson_network(
        Av: float = 2,
        G0: float = 1.7,
        shield: float = 1,
        n_h: float = 611,
        T: float = 10
    ):

    return ReactionNetwork(14, n_h,
        Reaction([0], [1,2], 1.2e-17),
        Reaction([3], [4,2], 6.8e-18),
        Reaction([1,5], [6,0], 2e-9),
        Reaction([1,7], [8,0], 8e-10),
        Reaction([1,9], [10,0], 1.7e-9),
        Reaction([1,0], [3], 7e-15),
        Reaction([4,9], [11,7,3], 1.6e-9),
        Reaction([11,0], [6], 4e-16),
        Reaction([11,8], [10], 1e-9),
        Reaction([7,6], [9], 2e-10),
        Reaction([5,8], [9], 5.8e-12*T**0.5),
        Reaction([4,2], [3], 9e-11/T**0.64),
        Reaction([1,2], [0], 1.9e-6/T**0.54),
        Reaction([11,2], [5], 1.4e-10/T**0.61),
        Reaction([10,2], [9], 3.3e-5/T),
        Reaction([12,2], [13], 3.8e-10/T**0.65),
        Reaction([1,13], [12,2,0], 2e-9),
        Reaction([5], [11,2], 3e-10*G0*np.exp(-3*Av)),
        Reaction([6], [5], 1e-9*G0*np.exp(-1.5*Av)),
        Reaction([9], [5,7], 1e-10*shield*G0*np.exp(-3*Av)),
        Reaction([8], [7], 5e-10*G0*np.exp(-1.7*Av)),
        Reaction([13], [12,2], 2e-10*G0*np.exp(-1.9*Av)),
        Reaction([10], [9], 1.5e-10*G0*np.exp(-2.5*Av))
    )

