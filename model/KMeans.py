import numpy as np
from utils import euclidean_dist_squared
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        self.means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            self.means[kk] = X[i]

        while True:
            y_old = y

            dist2 = euclidean_dist_squared(X, self.means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            for kk in range(self.k):
                if np.any(y==kk):
                    self.means[kk] = X[y==kk].mean(axis=0)

            changes = np.sum(y != y_old)

            if changes == 0:
                break

    def predict(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        N, D = X.shape

        y_pred = self.predict(X)
        retVal = 0

        for i in range(N):
            for j in range(D):
                retVal += (X[i][j] - self.means[y_pred[i]][j]) ** 2

        return retVal
