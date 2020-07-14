import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Binary Classification
    df = pd.read_csv('data/data_banknote_authentication.csv', )
    X = df.values[:,:3]
    y = df.values[:,4]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Multi-Classification
    df = pd.read_csv('data/processed_cleveland.csv', )
    X = df.values[:,:12]
    y = df.values[:,13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
