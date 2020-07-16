import numpy as np


class NaiveBayes:
    def __init__(self, num_classes, beta=0):
        self.num_classes = num_classes
        self.beta = beta

    def fit(self, X, y):
        N, D = X.shape
        C = self.num_classes
        counts = np.bincount(y)
        p_y = counts / N
        p_xy = np.zeros((D, C))

        for c in range(C):
            X_c = X[y == c]
            for d in range(D):
                p_xy[d][c] = (np.sum(X_c[:,d]) + self.beta) / (counts[c] + 2 * self.beta)

        self.p_y = p_y
        self.p_xy = p_xy

    def predict(self, X):
        N, D = X.shape
        C = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y
        y_pred = np.zeros(N)

        for n in range(N):
            probs = p_y.copy()
            for d in range(D):
                if X[n, d] != 0:
                    probs *= p_xy[d, :]
                else:
                    probs *= (1-p_xy[d, :])
            y_pred[n] = np.argmax(probs)

        return y_pred


import os
import pickle

if __name__ == '__main__':
    with open(os.path.join('..', 'data', 'newsgroups.pkl'), 'rb') as f:
        dataset = pickle.load(f)

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    model = NaiveBayes(4, beta=1)
    model.fit(X, y)

    print("training error: %.3f" %np.mean(model.predict(X) != y))
    print("validation error: %.3f" %np.mean(model.predict(X_valid) != y_valid))
