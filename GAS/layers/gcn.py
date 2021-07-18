# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = self.activation(act)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def activation(self, act):
        if act == 'relu':
            return nn.ReLU()
        elif act == 'prelu':
            return nn.PReLU()
        elif act == 'rrelu':
            return nn.RReLU()
        elif act == 'lrelu':
            return nn.LeakyReLU(negative_slope=0.1)
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softplus':
            return nn.Softplus()
        elif act == 'relu6':
            return nn.ReLU6()
        elif act == 'selu':
            return nn.SELU()
        else:
            return None

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        if self.act is not None:
            out = self.act(out)
        return out


