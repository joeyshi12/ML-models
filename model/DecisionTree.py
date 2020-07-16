import numpy as np
from scipy import stats


def gini(p):
    return np.sum(p * (1 - p))


def entropy(p):
    plogp = 0*p # initialize full of zeros
    plogp[p>0] = p[p>0]*np.log(p[p>0]) # only do the computation when p>0
    return -np.sum(plogp)


class DecisionTree:
    """Decision tree classifier model"""
    def __init__(self, max_depth, criterion='gini'):
        self.max_depth = max_depth
        self.criterion = criterion
        if criterion == 'gini':
            self.impurity = gini
        elif criterion == 'entropy':
            self.impurity = entropy
        else:
            raise ValueError

    def getSplitParameters(self, X, y):
        n, d = X.shape
        count = np.bincount(y)
        y_mode = np.argmax(count)
        p = count / np.sum(count);
        impurityBefore = self.impurity(p)
        maxGain = 0

        rightLabel = y_mode
        leftLabel = None
        splitCol = None
        splitVal = None

        if np.unique(y).size <= 1:
            return rightLabel, leftLabel, splitCol, splitVal

        for j in range(d):
            for i in range(n):
                value = X[i, j]
                y_right = y[X[:, j] > value]
                y_left = y[X[:, j] <= value]
                count_right = np.bincount(y_right, minlength=count.size)
                count_left = count - count_right

                p_right = count_right if np.sum(count_right) == 0 else count_right / np.sum(count_right)
                p_left = count_left if np.sum(count_left) == 0 else count_left / np.sum(count_left)
                impurityAfter = (y_right.size / n) * self.impurity(p_right) + (y_left.size / n) * self.impurity(p_left)
                gain = impurityBefore - impurityAfter

                if gain > maxGain:
                    maxGain = gain
                    rightLabel = stats.mode(y_right.flatten())[0][0]
                    leftLabel = stats.mode(y_left.flatten())[0][0]
                    splitCol = j
                    splitVal = value

        return rightLabel, leftLabel, splitCol, splitVal

    def fit(self, X, y):
        """
        Trains decision tree to X and y
        X: nxd matrix
        y: n-dimensional vector of labels 0,1,...,c
        """
        rightLabel, leftLabel, splitCol, splitVal = self.getSplitParameters(X, y)

        if self.max_depth <= 1 or splitCol is None:
            self.rightChild = None
            self.leftChild = None
            self.rightLabel = rightLabel
            self.leftLabel = leftLabel
            self.splitCol = splitCol
            self.splitVal = splitVal
            return

        self.splitCol = splitCol
        self.splitVal = splitVal
        rightIdx = X[:, splitCol] > splitVal
        leftIdx = X[:, splitCol] <= splitVal

        self.rightChild = DecisionTree(self.max_depth-1, criterion=self.criterion)
        self.rightChild.fit(X[rightIdx], y[rightIdx])
        self.leftChild = DecisionTree(self.max_depth-1, criterion=self.criterion)
        self.leftChild.fit(X[leftIdx], y[leftIdx])

    def predict(self, X):
        m, d = X.shape
        yhat = np.zeros(m)

        if self.splitVal is None:
            yhat = self.rightLabel * np.ones(m)

        elif self.rightChild is None:
            for i in range(m):
                if X[i, self.splitCol] > self.splitVal:
                    yhat[i] = self.rightLabel
                else:
                    yhat[i] = self.leftLabel

        else:
            rightIdx = X[:, self.splitCol] > self.splitVal
            leftIdx = X[:, self.splitCol] <= self.splitVal
            yhat[rightIdx] = self.rightChild.predict(X[rightIdx])
            yhat[leftIdx] = self.leftChild.predict(X[leftIdx])

        return yhat



import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
        dataset = pickle.load(f)

    X = dataset["X"]
    y = dataset["y"]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

    depths = np.arange(1, 13)
    train_errors = np.zeros(12)
    test_errors = np.zeros(12)

    for depth in depths:
        model = DecisionTree(max_depth=depth, criterion='entropy')
        model.fit(Xtrain, ytrain)
        train_errors[depth - 1] = np.sum(model.predict(Xtrain) != ytrain) / X.shape[0]
        test_errors[depth - 1] = np.sum(model.predict(Xtest) != ytest) / X.shape[0]

    plt.plot(depths, train_errors, label="training error")
    plt.plot(depths, test_errors, label="testing error")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(os.path.join("..", "fig", "decision_tree_errors.pdf"))
