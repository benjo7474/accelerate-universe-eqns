import time
from types import NoneType
import numpy as np
import faiss
import matplotlib.pyplot as plt

# TODO eventually make labels multi-d. This would make labels a matrix and gradients a rank-3 tensor.


# Contains machinery that takes in features
# Does nearest neighbors searches via FAISS
# Includes the option to do gradient boosting via a Taylor series expansion
class GradientBoostModel:

    # CONSTRUCTOR
    def __init__(self, features: np.ndarray, labels: np.ndarray, gradients: np.ndarray = None):
        
        # --- INPUT ERROR CHECKING ---

        # check that length of vectors is correct
        self._mf, self._nf = features.shape
        self._ml           = len(labels)

        if type(gradients) != NoneType:
            self._mg, self._ng = gradients.shape
            if self._nf != self._ng:
                raise ValueError('Number of features does not match length of each gradient vector')
            if self._mf != self._mg:
                raise ValueError('Features and gradients has a mismatching size of points')
        if self._mf != self._ml:
            raise ValueError('Features and labels has a mismatching size of points')
        

        # save data
        self.features = features
        self.labels = labels
        self.gradients = gradients

        # build FAISS index
        self._faiss_index = faiss.IndexFlatL2(self._nf)
        self._faiss_index.add(features)


    # given a matrix of features, uses FAISS to return the predicted value.
    # uses the average of the k-nearest neighbors (default only uses 1 nearest neighbor)
    def predict(self, targets: np.ndarray, k: int, unwrap_log=[False,False,False], disp_time=False):

        start_time = time.perf_counter()
        # TODO check if the length of each feature vector matches the length of training vectors

        # use FAISS to do k-nearest neighbors search
        _, I = self._faiss_index.search(targets, k)
        # compute distances from source to target features
        
        # if necessary we deal with distances by unwrapping log
        features_unlog = self.features.copy()
        targets_unlog = targets.copy()
        for j, val in enumerate(unwrap_log):
            if val == True:
                targets_unlog[:,j] = 10 ** targets_unlog[:,j]
                features_unlog[:,j] = 10 ** features_unlog[:,j]
        
        predictions = np.zeros(len(targets))
        for j in range(k):
            if type(self.gradients) == NoneType:
                # gradient not specified; use only nearest neighbor
                predictions += self.labels[I[:,j]]
            else:
                distance_vecs = targets_unlog - features_unlog[I[:,j]]
                predictions += self.labels[I[:,j]] + np.sum(self.gradients[I[:,j]] * distance_vecs, axis=1)

        if disp_time == True:
            end_time = time.perf_counter()
            print(f'Time elapsed for predictions: {end_time-start_time} seconds')

        # return average
        return predictions / k
    
