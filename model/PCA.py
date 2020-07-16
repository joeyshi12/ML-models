import numpy as np

class PCA:
    """PCA model with squared recontruction loss"""

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mu = np.mean(X,axis=0)
        X = X - self.mu

        U, s, Vh = np.linalg.svd(X)
        self.W = Vh[:self.k]

    def compress(self, X):
        X = X - self.mu
        Z = X@self.W.T
        return Z

    def expand(self, Z):
        X = Z@self.W + self.mu
        return X
