import numpy as np


class LeastSquares:
    """Ridge regression model"""
    def __init__(self, fit_intercept=False, lammy=1):
        self.fit_intercept = fit_intercept
        self.lammy = lammy

    def fit(self,X,y):
        if self.fit_intercept:
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        self.w = solve(X.T @ X + self.lammy * np.eye(X.shape[1]), X.T @ y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        return X @ self.w


def kernel_RBF(X1, X2, sigma=1):
    return np.exp(-utils.euclidean_dist_squared(X1, X2) / (2 * sigma ** 2))


def kernel_poly(X1, X2, p=2):
    return (1 + X1 @ X2.T) ** p


def kernel_linear(X1, X2):
    return X1 @ X2.T


class KernelLeastSquares:
    """L2 regularized least squares model with a change in basis"""
    def __init__(self, lammy=1, kernel_fun=kernel_linear, **kernel_args):
        self.lammy = lammy
        self.kernel_fun = kernel_fun
        self.kernel_args = kernel_args

    def fit(self,X,y):
        self.X = X
        K = self.kernel_fun(X, X, **self.kernel_args)
        self.u = np.linalg.inv(K + self.lammy * np.eye(K.shape[0])) @ y

    def predict(self, X):
        Ktest = self.kernel_fun(Xtest, self.X, **self.kernel_args)
        return Ktest @ self.u
