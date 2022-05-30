import numpy as np
import h5py
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load_toxic_bert():
    train_h5 = h5py.File("data/train_bert.h5")
    test_h5 = h5py.File("data/test_bert.h5")
    X_train, A_train, Y_train = (
        np.array(train_h5["X"]),
        np.array(train_h5["asian_or_black"]),
        np.array(train_h5["Y"]),
    )
    X_test, A_test, Y_test = (
        np.array(test_h5["X"]),
        np.array(test_h5["asian_or_black"]),
        np.array(test_h5["Y"]),
    )
    return X_train, X_test, A_train, A_test, Y_train, Y_test


if __name__ == "__main__":



    X_train, X_test, A_train, A_test, y_train, y_test = load_toxic_bert()
    print(X_train.shape, A_train.shape, y_train.shape)
    print(X_test.shape, A_test.shape, y_test.shape)
