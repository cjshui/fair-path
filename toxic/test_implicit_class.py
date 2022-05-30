import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from dataset import load_toxic_bert
from model import Fea, Clf
from utils import  train_implicit, evaluate_pp_implicit



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

for iter in range(1):
    # get train/test data
    X_train, X_test, A_train, A_test, y_train, y_test = load_toxic_bert()

    ### convert Y to binary with threshold=0
    y_train = np.array(y_train > 0, dtype=np.int)
    y_test = np.array(y_test > 0, dtype=np.int)

    # initialize model
    fea = Fea(input_size=len(X_train[0])).cuda()
    clf_0 = Clf().cuda()
    clf_1 = Clf().cuda()


    optim_fea = optim.Adam(fea.parameters(), lr=1e-3, eps=1e-3)
    optim_clf_0 = optim.Adam(clf_0.parameters(), lr=1e-3, eps=1e-3)
    optim_clf_1 = optim.Adam(clf_1.parameters(), lr=1e-3, eps=1e-3)

    criterion = nn.BCELoss()

    train_implicit(fea, clf_0, clf_1, criterion, optim_fea, optim_clf_0, optim_clf_1, X_train, A_train, y_train, kappa=3e-4,max_inner=15,out_step=5)


    ap_test, gap_test = evaluate_pp_implicit(fea, clf_0, clf_1, X_test, y_test, A_test)

    print("The accuracy is:", ap_test)
    print("The prediction gap is:", gap_test)
