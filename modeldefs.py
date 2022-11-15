import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

import time

class FCN(nn.Module):
    def __init__(self, in_feats, hidden_neurons=[128, 64],
                 out_feats=16, device='cpu'):
        super(FCN, self).__init__()
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(in_feats, hidden_neurons[0]))
        for i in range(1, len(hidden_neurons)):
            self.linear_layers.append(nn.Linear(hidden_neurons[i-1], hidden_neurons[i]))
        self.linear_layers.append(nn.Linear(hidden_neurons[-1], out_feats))

    def forward(self,features):
        fh = features.clone()
        for i, layer in enumerate(self.linear_layers):
            if i == (len(self.linear_layers)-1):
                fh = layer(fh)
            else:
                fh = F.relu(layer(fh))
        return fh