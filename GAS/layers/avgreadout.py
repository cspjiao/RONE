# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

# Applies an average on seq, of shape (batch, nodes, features)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)
        


