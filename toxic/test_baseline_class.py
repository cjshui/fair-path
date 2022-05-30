import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataset import load_toxic_bert
from model import Net
from utils import train_dp, evaluate_dp, evaluate_pp_model


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# get train/test data
X_train, X_test, A_train, A_test, y_train, y_test = load_toxic_bert()

### convert Y to binary with threshold=0
y_train = np.array(y_train > 0, dtype=np.int)
y_test = np.array(y_test > 0, dtype=np.int)

model = Net(input_size=len(X_train[0])).cuda()
optimizer = optim.Adam(model.parameters(), lr=3e-3, eps=1e-3)
criterion = nn.BCELoss()
lam = 0.5


method='erm'
train_dp(model, criterion, optimizer, X_train, A_train, y_train, method, lam, batch_size=500, niter=100)
ap_test, gap_test = evaluate_pp_model(model, X_test, y_test, A_test)
print("The accuracy is:", ap_test)
print("The prediction gap is:", gap_test)
