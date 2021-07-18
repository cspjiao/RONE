# coding=utf-8
"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv
import numpy as np


class GCNEncoder(nn.Module):
    def __init__(self, G, config, activation=F.tanh, dropout=0.5):
        super(GCNEncoder, self).__init__()
        G = DGLGraph(G)
        self.G = G
        self.config = config
        self.encoder = GraphConv(self.config.struct[0], self.config.struct[-1],
                                 activation=activation).to(
            self.config.device, dtype=torch.float32)
        self.encoder2 = GraphConv(self.config.struct[-1], self.config.struct[-1],
                                  activation=activation).to(
            self.config.device, dtype=torch.float32)
        self.enc_mu = GraphConv(self.config.struct[-1], self.config.struct[-1],
                                activation=activation).to(
            self.config.device, dtype=torch.float32)
        self.enc_log_sigma = GraphConv(self.config.struct[-1], self.config.struct[-1],
                                       activation=activation).to(
            self.config.device, dtype=torch.float32)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = self.encoder(self.G, features)
        h = self.dropout(self.encoder2(self.G, h))
        # h = self.dropout(h)
        mu = self.dropout(self.enc_mu(self.G, h))
        log_sigma = self.dropout(self.enc_log_sigma(self.G, h))
        sigma = log_sigma.exp()
        z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).to(
            self.config.device, dtype=torch.float32)
        z = mu + sigma * z
        return mu, sigma, z
