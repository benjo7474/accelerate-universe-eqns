import numpy as np
import faiss
import matplotlib.pyplot as plt

# TODO add option to specify k for k-nearest neighbors, then take average of the predictions.

# TODO eventually make labels multi-d. This would make labels a matrix and gradients a rank-3 tensor.
# for this endeavor is it better to use one pandas dataframe and specify the columns to use?

# Contains machinery that takes in features
# Does nearest neighbors searches via FAISS
# Includes the option to do gradient boosting via a Taylor series expansion
class GradientBoostModel:

    # CONSTRUCTOR
    def __init__(self, features: np.ndarray, labels: np.ndarray, gradients: np.ndarray = None):
        
        # INPUT ERROR CHECKING
        # cannot have any of the features be undefined
        # maybe don't worry about this for now...

        # check that length of vectors is correct
        self._mf, self._nf = features.shape
        self._ml, self._nl = labels.shape
        self._mg, self._ng = gradients.shape

        if self._mf != self._ml or self._mf != self._mg:
            raise ValueError('ERROR: one of features, labels or gradients has a mismatching size of points')
        if self._nf != self._ng:
            raise ValueError('ERROR: number of features does not match length of each gradient vector')

        # save data
        self.features = features
        self.labels = labels
        self.gradients = gradients

        # build FAISS index
        self._faiss_index = faiss.IndexFlatL2(self._nf)
        self._faiss_index.add(features)


    # given a matrix of features, uses FAISS to return the predicted value.
    # uses the average of the k-nearest neighbors (default only uses 1 nearest neighbor)
    def predict(self, targets: np.ndarray):

        # TODO check if the length of each feature vector matches the length of training vectors

        # use FAISS to do k-nearest neighbors search
        D, I = self._faiss_index.search(targets, 1)
        # compute distances from source to target features
        distance_vecs = targets - self.features[I]
        
        # gradient not specified; use only nearest neighbor
        if self.gradients == None:
            predictions = self.labels[I]
        else:
        # gradient specified; use formula q(x) ~= q(x_0) + grad(q(x_0)) * (x-x_0)
            predictions = self.labels[I] + np.sum(self.gradients[I] * distance_vecs, axis=1)

        return predictions
    

    # given a matrix of features and a list of true values, use above function to predict,
    # followed by comparing the accuracy to true_values and displaying error statistics.
    def test_accuracy(self, targets: np.ndarray, true_values: np.ndarray):

        # TODO check if length of targets and true_values matches
        # TODO check if length of each feature vector matches length of training vectors

        predicted_values = self.predict(targets, 1)
        absolute_error = np.abs(predicted_values - true_values)
        percent_error = self.absolute_error / np.abs(true_values)

        print('PERCENT ERRORS')
        print(f'Mean: {np.mean(percent_error)}')
        print(f'Median: {np.median(percent_error)}')
        print(f'Max: {np.max(percent_error)}')
        print(f'STD: {np.std(percent_error)}')
        tol = 0.05
        num_vals_outside_error = (percent_error > tol).sum()
        print(f'# of points outside {tol*100}% error: {num_vals_outside_error} ({num_vals_outside_error/len(percent_error)}% of data)\n')

        plt.figure()
        plt.hist(percent_error, bins=np.arange(0, 0.3, 0.005))
        plt.title('Percent Error')

        print('ABSOLUTE ERRORS')
        print(f'Mean: {np.mean(absolute_error)}')
        print(f'Median: {np.median(absolute_error)}')
        print(f'Max: {np.max(absolute_error)}')
        print(f'STD: {np.std(absolute_error)}')

        plt.figure()
        plt.hist(np.log10(absolute_error))
        plt.title('Absolute Error')

