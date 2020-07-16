import numpy as np
from scipy import stats
import utils
import timeit

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, Xtest):
        N, D = Xtest.shape

        dist_squared = utils.euclidean_dist_squared(Xtest, self.X)
        y_pred = np.zeros(N)

        for i in range(N):
            d = dist_squared[i]

            sorted_indices = np.argsort(d)
            nearest_k = self.y[sorted_indices[:self.k]]

            y_pred[i] = utils.mode(nearest_k)

        return y_pred
