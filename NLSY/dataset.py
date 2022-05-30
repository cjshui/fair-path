from numpy.lib.npyio import load
import pandas as pd
import numpy as np
import os.path as osp

from sklearn.model_selection import train_test_split


def load_nlsy79(test_size=0.2):
    df = pd.read_csv("nlsy79-datasets/data-processed.csv")
    Y = df["income"]
    A = df["gender"]
    X = df.drop(["income", "gender"], axis=1)
    # print("#a=0", np.count_nonzero(A == 0))
    # print("#a=1", np.count_nonzero(A == 1))
    X_train, X_test, A_train, A_test, y_train, y_test = train_test_split(
        X, A, Y, test_size=test_size
    )
    return (
        X_train.to_numpy(),
        X_test.to_numpy(),
        A_train.to_numpy(),
        A_test.to_numpy(),
        y_train.to_numpy(dtype=float),
        y_test.to_numpy(dtype=float),
    )


if __name__ == "__main__":
    X_train, X_test, A_train, A_test, y_train, y_test = load_nlsy79(test_size=0.2)
    print(X_train.shape)
