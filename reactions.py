from typing import Callable
import numpy as np
import torch
from scipy.integrate import solve_ivp

class Reaction:
    # Represents a reaction of course

    # Constructor: takes rate (as a function of parameters vector), and reactants and products as a list of numbers
    def __init__(self, reactants, products, rate: Callable):
        self.rate = rate
        self.reactants = reactants
        self.products = products
        self.N_reactants = np.size(reactants)
        self.N_products = np.size(products)
        if len(reactants) == 2:
            self.rate_mod = lambda p: rate(p) * p[0]
        elif len(reactants) == 1:
            self.rate_mod = lambda p: rate(p) * torch.tensor([1])

    # Now build a method that converts this info to entries in a torch sparse coo tensor
    def build_tensor_structures(self, params):
        N = self.N_reactants + self.N_products
        if self.N_reactants == 1:
            indices_L = np.zeros(shape=(2,N))
            react1 = self.reactants[0]
            indices_L[:,0] = np.array([react1, react1])
            for j in range(self.N_products):
                indices_L[:,j+1] = np.array([self.products[j], react1])
            values_L =  self.rate_mod(params) * np.ones(N)
            values_L[0] *= -1
            return indices_L, values_L
        elif self.N_reactants == 2:
            indices_Q = np.zeros(shape=(3,N))
            react1 = self.reactants[0]
            react2 = self.reactants[1]
            indices_Q[:,0] = np.array([react1, react1, react2])
            indices_Q[:,1] = np.array([react2, react1, react2])
            for j in range(self.N_products):
                indices_Q[:,j+2] = np.array([self.products[j], react1, react2])
            values_Q = self.rate_mod(params) * np.ones(N)
            values_Q[0] *= -1
            values_Q[1] *= -1
            return indices_Q, values_Q
        else:
            print('The Reaction class is not built for reactions involving more than two reactants.')

    def build_derivative_tensor_structures(self, params):

        if type(params) == np.ndarray:
            jacobian = torch.autograd.functional.jacobian(self.rate_mod, torch.from_numpy(params)).numpy().flatten()
        elif type(params) == torch.Tensor:
            jacobian = torch.autograd.functional.jacobian(self.rate_mod, params).numpy().flatten()

        N = self.N_reactants + self.N_products

        if self.N_reactants == 1:
            indices_dL = np.zeros(shape=(3,N*len(jacobian)))
            react1 = self.reactants[0]
            indices_dL[:,[0,1,2]] = np.array([[react1, react1, react1],
                                              [react1, react1, react1],
                                              [0,      1,      2     ]])
            for j in range(self.N_products):
                prod_j = self.products[j]
                indices_dL[:,[3*(j+1),3*(j+1)+1,3*(j+1)+2]] = np.array([[prod_j, prod_j, prod_j],
                                                                       [react1, react1, react1],
                                                                       [0,      1,      2     ]])
            values_dL = np.tile(jacobian, N)
            values_dL[0:3] *= -1
            return indices_dL, values_dL
        
        elif self.N_reactants == 2:
            indices_dQ = np.zeros(shape=(4,N*len(jacobian)))
            react1 = self.reactants[0]
            react2 = self.reactants[1]
            indices_dQ[:,0:6] = np.array([
                [react1, react1, react1, react2, react2, react2],
                [react1, react1, react1, react1, react1, react1],
                [react2, react2, react2, react2, react2, react2],
                [0,      1,      2,      0,      1,      2     ]
            ])
            for j in range(self.N_products):
                prod_j = self.products[j]
                indices_dQ[:,[3*(j+2),3*(j+2)+1,3*(j+2)+2]] = np.array([
                    [prod_j, prod_j, prod_j],
                    [react1, react1, react1],
                    [react2, react2, react2],
                    [0,      1,      2,    ]
                ])
            values_dQ = np.tile(jacobian, N)
            values_dQ[0:6] *= -1
            return indices_dQ, values_dQ
        else:
            print('The Reaction class is not built for reactions involving more than two reactants.')


class ReactionNetwork:
    # Holds a whole bunch of reactions

    # Constructor: takes in a bunch of reactions
    # Assumes first parameter in params vec is the normalizing density (log(n_h))
    # Reactions rate functions will take params as an input
    # As for which initial conditions these represent, this is encoded in QoIs_to_vary.
    def __init__(self,
        num_species,
        params,
        *reactions: Reaction,
        compute_sensitivities = False,
        QoIs_to_vary = np.array([])
    ):

        self.reactions = reactions
        self.num_species = num_species
        self.params = params
        self.compute_sensitivities = compute_sensitivities
        self.QoIs_to_vary = QoIs_to_vary
        self._build_tensors(params) # routine that saves Q and L, and if necessary, dQ/dp and dL/dp


    def num_nonzero_entries(self):
        N_L = 0
        N_Q = 0
        for reaction in self.reactions:
            if reaction.N_reactants == 1:
                N_L += 1 + reaction.N_products
            elif reaction.N_reactants == 2:
                N_Q += 2 + reaction.N_products
        return N_Q, N_L

    def _build_tensors(self, params):
        N_Q, N_L = self.num_nonzero_entries()
        n = self.num_species
        indices_Q = np.empty((3,N_Q))
        values_Q = np.empty(N_Q)
        indices_L = np.empty((2,N_L))
        values_L = np.empty(N_L)
        j_Q, j_L = 0, 0

        if self.compute_sensitivities == True:
            indices_dQ = np.empty((4,N_Q*len(params)))
            values_dQ = np.empty(N_Q*len(params))
            indices_dL = np.empty((3,N_L*len(params)))
            values_dL = np.empty(N_L*len(params))

        for reaction in self.reactions:
            indices, values = reaction.build_tensor_structures(params)
            N = reaction.N_reactants + reaction.N_products
            if reaction.N_reactants == 1:
                indices_L[:,j_L:j_L+N] = indices
                values_L[j_L:j_L+N] = values
                if self.compute_sensitivities == True:
                    indices, values = reaction.build_derivative_tensor_structures(params)
                    indices_dL[:,len(params)*j_L:len(params)*(j_L+N)] = indices
                    values_dL[len(params)*j_L:len(params)*(j_L+N)] = values
                j_L += N
            elif reaction.N_reactants == 2:
                indices_Q[:,j_Q:j_Q+N] = indices
                values_Q[j_Q:j_Q+N] = values
                if self.compute_sensitivities == True:
                    indices, values = reaction.build_derivative_tensor_structures(params)
                    indices_dQ[:,len(params)*j_Q:len(params)*(j_Q+N)] = indices
                    values_dQ[len(params)*j_Q:len(params)*(j_Q+N)] = values
                j_Q += N
        
        self.Q = torch.sparse_coo_tensor(indices=indices_Q, values=values_Q, size=(n,n,n), dtype=torch.float64)
        self.L = torch.sparse_coo_tensor(indices=indices_L, values=values_L, size=(n,n), dtype=torch.float64)
        self.Q_dense = self.Q.to_dense()

        if self.compute_sensitivities == True:
            self.dQ_dp = torch.sparse_coo_tensor(indices=indices_dQ, values=values_dQ, size=(n,n,n,len(params)), dtype=torch.float64).to_dense()
            self.dL_dp = torch.sparse_coo_tensor(indices=indices_dL, values=values_dL, size=(n,n,len(params)), dtype=torch.float64).to_dense()
        
    
    def reaction_rhs(self, t, x: np.ndarray) -> np.ndarray:
        # takes a vector from numpy and returns vector from numpy
        xt = torch.from_numpy(x)
        return self.reaction_rhs_torch(xt).numpy()

    def reaction_rhs_torch(self, x: torch.tensor) -> torch.tensor:
        # takes a vector from torch and does computation
        Q2 = torch.tensordot(self.Q_dense, x, dims=([2],[0]))
        return torch.linalg.matmul(Q2 + self.L, x)

    def jacobian_torch(self, x: torch.tensor) -> torch.tensor:
        # takes a vector from torch, returns torch
        return torch.tensordot(self.Q_dense, x, dims=([1],[0])) + torch.tensordot(self.Q_dense, x, dims=([2],[0])) + self.L

    def jacobian(self, t, x: np.ndarray) -> np.ndarray:
        # takes vector from numpy, returns numpy
        xt = torch.from_numpy(x)
        return self.jacobian_torch(xt).numpy()
    
    def jacobian_sensitivities_torch(self, x: torch.tensor) -> torch.tensor:
        # takes vector from torch, returns torch (sparse)
        jac_block = self.jacobian_torch(x[:self.num_species])
        n_p = len(self.params) + len(self.QoIs_to_vary)
        crow_indices = torch.arange(n_p+2)
        col_indices = torch.arange(n_p+1)
        values = jac_block.tile((n_p+1,1,1))
        return torch.sparse_bsr_tensor(crow_indices, col_indices, values)

    def jacobian_sensitivities(self, t, x: np.ndarray) -> np.ndarray:
        # takes vector from numpy, returns numpy
        xt = torch.from_numpy(x)
        return self.jacobian_sensitivities_torch(xt).to_dense().numpy()
    
    # TODO: modify this function to contain sensitivities for 
    def sensitivities_rhs_torch(self, x: torch.tensor) -> torch.tensor:
        # takes a vector from torch and does computation
        f = torch.zeros(len(x))
        N = self.num_species
        Np = len(self.params)
        Nxv = len(self.QoIs_to_vary)

        # the first entries of x are the normal rhs of the reaction.
        species = x[:N]
        f[:N] = self.reaction_rhs_torch(species)
        jacobian = self.jacobian_torch(species)
        # every other entry is the derivative. Go parameter by parameter
        for j in range(Np):
            l = (j + 1) * N
            u = (j + 2) * N
            s = x[l:u]
            f[l:u] = torch.linalg.matmul(jacobian, s) + \
                    torch.linalg.matmul(torch.tensordot(self.dQ_dp[:,:,:,j], species, dims=([2],[0])) + self.dL_dp[:,:,j], species)
        for j in range(Np, Np + Nxv):
            l = (j + 1) * N
            u = (j + 2) * N
            s = x[l:u]
            f[l:u] = torch.linalg.matmul(jacobian, s)
        
        return f
    
    def sensitivities_rhs(self, t, x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(x)
        return self.sensitivities_rhs_torch(xt).numpy()
    

    def solve_reaction(self,
        t_range,
        x0,
        t_eval = None,
    ):

        # do not compute sensitivities; no need for anything fancy
        if self.compute_sensitivities == False:

            soln = solve_ivp(self.reaction_rhs, t_range, x0, method='LSODA',
                rtol=1e-10,
                jac=self.jacobian,
                max_step=(t_range[1]-t_range[0])/1e3,
                first_step=0.25,
                t_eval=t_eval)
            
        # If we want to compute sensitivities, we need to know which initial conditions to include.
        # Assume that every physical parameter is always included.
        elif self.compute_sensitivities == True:

            # pad the initial condition with zeros
            zeros = np.zeros(self.num_species)
            x0_padded = x0
            for i in range(len(self.params)):
                x0_padded = np.concatenate([x0_padded, zeros])

            # depending on what initial conditions we are varying,
            # we need to include some ones in the correct places.
            for i in range(len(self.QoIs_to_vary)):
                zeros = np.zeros(self.num_species)
                zeros[self.QoIs_to_vary[i]] = 1
                x0_padded = np.concatenate([x0_padded, zeros])

            soln = solve_ivp(self.sensitivities_rhs, t_range, x0_padded, method='LSODA',
                rtol=1e-10,
                jac=self.jacobian_sensitivities,
                max_step=(t_range[1]-t_range[0])/1e3,
                first_step=0.25,
                t_eval=t_eval)
        return soln.t, soln.y


    def solve_reaction_snapshot(self, x0, tf, QoI):
        if self.compute_sensitivities == False:
            soln = solve_ivp(self.reaction_rhs, [0, tf], x0, method='LSODA',
                rtol=1e-10,
                jac=self.jacobian,
                max_step=tf/1e3,
                first_step=0.25,
                t_eval=[tf])
        elif self.compute_sensitivities == True:
            # pad the initial condition with zeros
            zeros = np.zeros(self.num_species)
            x0_padded = x0
            for i in range(len(self.params)):
                x0_padded = np.concatenate([x0_padded, zeros])
            soln = solve_ivp(self.sensitivities_rhs, tf, x0_padded, method='LSODA',
                rtol=1e-10,
                jac=self.jacobian_sensitivities,
                max_step=tf/1e3,
                first_step=0.25,
                t_eval=[tf])
        return soln.y.flatten()[QoI]

        