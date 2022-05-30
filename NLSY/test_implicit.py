import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataset import load_nlsy79
from model import Fea, Reg
from utils import train_implicit, evaluate_pp_implicit



for iter in range(1):

    X_train, X_test, A_train, A_test, y_train, y_test = load_nlsy79()

    # initialize model
    fea = Fea(input_size=len(X_train[0])).cuda()
    reg_0 = Reg().cuda()
    reg_1 = Reg().cuda()

    optim_fea = optim.Adam(fea.parameters(),lr=1e-3,eps=1e-3)
    optim_reg_0 = optim.Adam(reg_0.parameters(),lr=1e-3,eps=1e-3)
    optim_reg_1 = optim.Adam(reg_1.parameters(),lr=1e-3,eps=1e-3)
    criterion = nn.MSELoss()

    train_implicit(fea,reg_0,reg_1,criterion,optim_fea,optim_reg_0,optim_reg_1,X_train,A_train,y_train,kappa=2e-3)
    ap_test, gap_test, _, _  =  evaluate_pp_implicit(fea, reg_0,reg_1, X_test, y_test, A_test)
    print("The mse is:", ap_test)
    print("The prediction gap is:",gap_test)
