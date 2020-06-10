#####################################
# models.py
# contains the definitions for the target neural network architectures
#
####################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneLayerDNN(nn.Module):
    def __init__(self, d_in, d_out):
        super(OneLayerDNN, self).__init__()
        self.linear1 = nn.Linear(d_in, 2)

    def forward(self, x):
        x = self.linear1(x)
        return x


class TwoLayerDNN(nn.Module):
    def __init__(self, d_in, d_hidden):
        super(TwoLayerDNN, self).__init__()
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.linear2 = nn.Linear(d_hidden, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


