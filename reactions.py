import numpy as np
import torch


class Reaction:
    # Represents a reaction of course

    # Constructor: takes rate, and reactants and products as a list of numbers
    def __init__(self, reactants, products, rate):
        self.rate = rate
        self.reactants = reactants
        self.products = products
        self.N_reactants = np.size(reactants)
        self.N_products = np.size(products)

    # Now build a method that converts this info to entries in a torch sparse coo tensor
    def build_tensor_structures(self):
        N = self.N_reactants + self.N_products
        if self.N_reactants == 1:
            indices_L = np.zeros((N,2), dtype=np.int32)
            indices_L[0,:] = np.array([self.reactants[0], self.reactants[0]])
            for j in range(self.N_products):
                indices_L[j+1,:] = np.array([self.products[j], self.reactants[0]])
            values_L =  np.full(N, self.rate)
            values_L[0] *= -1
            return indices_L, values_L
        elif np.size(self.reactants) == 2:
            indices_Q = np.zeros((N,3), dtype=np.int32)
            indices_Q[0,:] = np.array([self.reactants[0], self.reactants[0], self.reactants[1]])
            indices_Q[1,:] = np.array([self.reactants[1], self.reactants[0], self.reactants[1]])
            for j in range(self.N_products):
                indices_Q[j+2,:] = np.array([self.products[j], self.reactants[0], self.reactants[1]])
            values_Q= np.full(N, self.rate)
            values_Q[0] *= -1
            values_Q[1] *= -1
            return indices_Q, values_Q
        else:
            print('The Reaction class is not built for reactions involving more than two reactants.')


class ReactionNetwork:
    # Holds a whole bunch of reactions

    # Constructor: takes in a bunch of reactions
    def __init__(self, num_species, *reactions: Reaction):
        self.reactions = reactions
        self.num_species = num_species

    def num_nonzero_entries(self):
        N_L = 0
        N_Q = 0
        for reaction in self.reactions:
            if reaction.N_reactants == 1:
                N_L += 1 + reaction.N_products
            elif reaction.N_reactants == 2:
                N_Q += 2 + reaction.N_products
        return N_Q, N_L

    def build_tensors(self):
        N_Q, N_L = self.num_nonzero_entries()
        indices_Q = np.empty((N_Q,3))
        values_Q = np.empty((N_Q))
        indices_L = np.empty((N_L,2))
        values_L = np.empty((N_L))
        j_Q, j_L = 0, 0
        for reaction in self.reactions:
            if reaction.N_reactants == 1:
                indices, values = reaction.build_tensor_structures()
                N = reaction.N_reactants + reaction.N_products
                indices_L[j_L:j_L+N,:] = indices
                values_L[j_L:j_L+N] = values
                j_L += N
            elif reaction.N_reactants == 2:
                indices, values = reaction.build_tensor_structures()
                N = reaction.N_reactants + reaction.N_products
                indices_Q[j_Q:j_Q+N,:] = indices
                values_Q[j_Q:j_Q+N] = values
                j_Q += N
        
        Q = torch.sparse_coo_tensor(indices=indices_Q.transpose().tolist(), values=values_Q.transpose().tolist(),
                                    size=(self.num_species, self.num_species, self.num_species), dtype=torch.float64)
        L = torch.sparse_coo_tensor(indices=indices_L.transpose().tolist(), values=values_L.transpose().tolist(),
                                    size=(self.num_species, self.num_species), dtype=torch.float64)
        return Q, L
    