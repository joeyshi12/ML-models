import numpy as np
from sklearn.preprocessing import LabelBinarizer


class NNClassifier:
    """Neural network classification model with softmax output layer"""
    def __init__(self, hidden_layer_sizes=[100], activation='relu', alpha=0.0001, lammy=1, batch_size=None, num_epochs=10, verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lammy = lammy

        if activation == 'relu':
            self.phi = lambda x: np.maximum(0, x)
            self.phi_deriv = lambda A: A > 0
        elif activation == 'sigmoid':
            self.phi = lambda x: 1 / (1 + np.exp(-x))
            self.phi_deriv = lambda A: A * (1 - A)
        else:
            raise ValueError

        self.alpha = alpha
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose

    def funObj(self, X, Y):
        activations = [X]
        for W, b in zip(self.weights, self.biases):
            Z = X @ W.T + b
            X = self.phi(Z)
            activations.append(X)

        # activation and activation derivative of output layer
        A = np.exp(Z) / np.sum(np.exp(Z), axis=1)[:,None]
        B = A * (1 - 1 / np.sum(np.exp(Z), axis=1)[:,None])

        loss = np.sum((A - Y) ** 2 / 2)
        # R = (activations[-1] - Y) * self.phi_deriv(activations[-1])
        R = (A - Y) * B
        # weight_grads = [R.T @ activations[-2] + self.lammy * self.weights[-1]]
        weight_grads = [R.T @ activations[-2] + self.lammy * self.weights[-1]]
        # bias_grads = [np.sum(R, axis=0) + self.lammy * self.biases[-1]]
        bias_grads = [np.sum(R, axis=0) + self.lammy * self.biases[-1]]

        for i in range(len(self.layer_sizes)-2, 0, -1):
            R = (R @ self.weights[i]) * self.phi_deriv(activations[i])
            weight_grads.insert(0, R.T @ activations[i-1] + self.lammy * self.weights[i-1])
            bias_grads.insert(0, np.sum(R, axis=0) + self.lammy * self.biases[i-1])

        return loss, weight_grads, bias_grads

    def fit(self, X, y):
        Y = LabelBinarizer().fit_transform(y)
        self.weights = list()
        self.biases = list()
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [Y.shape[1]]

        for i in range(len(self.layer_sizes)-1):
            self.weights.append(0.1 * np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]))
            self.biases.append(0.1 * np.random.randn(self.layer_sizes[i+1]))

        if self.batch_size is None:
            self.batch_size = np.min([200, X.shape[0]])

        for epoch in range(self.num_epochs):
            reordered_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
            for i in range(X.shape[0] // self.batch_size):
                batch = reordered_indices[i * self.batch_size : (i + 1) * self.batch_size]
                loss, weight_grads, bias_grads = self.funObj(X[batch], Y[batch])
                for i in range(len(self.layer_sizes)-1):
                    self.weights[i] -= self.alpha * weight_grads[i]
                    self.biases[i] -= self.alpha * bias_grads[i]
            if self.verbose:
                print("epoch %d, loss = %f" %(epoch+1, loss))

    def predict(self, X):
        for W, b in zip(self.weights, self.biases):
            Z = X @ W.T + b
            X = 1 / (1 + np.exp(-Z))
        return np.argmax(X, axis=1)


import os
import gzip
import pickle

if __name__ == '__main__':
    with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    X, y = train_set
    Xtest, ytest = test_set

    model = NNClassifier(hidden_layer_sizes=[256], alpha=0.0001, lammy=0.001, verbose=True)
    model.fit(X, y)
    print("training error: %.3f" %(np.sum(model.predict(X) != y) / X.shape[0]))
    print("testing error: %.3f" %(np.sum(model.predict(Xtest) != ytest) / Xtest.shape[0]))
