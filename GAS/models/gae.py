# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from layers import GCN

class GAE(nn.Module):
    def __init__(self, args, device):
        super(GAE, self).__init__()
        self.args = args
        self.device = device
        self.gcn1 = GCN(self.args.n_in, self.args.n_h1, self.args.act)
        self.gcn2 = GCN(self.args.n_h1, self.args.n_h2, self.args.act)
        self.dc1 = nn.Linear(self.args.n_h2, self.args.n_h3)
        self.dc2 = nn.Linear(self.args.n_h3, self.args.n_h4)
        self.act = self.activation(self.args.act)
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

    def encoder(self, X, adj):
        h1 = self.gcn1(X, adj)
        h2 = self.gcn2(h1, adj)
        return h2

    def decoder(self, X):
        h1 = self.dc1(X)
        if self.args.act is not None:
            h1 = self.act(h1)
        h2 = self.dc2(h1)
        if self.args.act is not None:
            h2 = self.act(h2)
        return h2

    def forward(self, X, adj, F):
        reps = self.encoder(X, adj)
        refts = self.decoder(reps)
        print(X.shape, refts.shape, F.shape)
        loss = torch.norm(refts - F)
        return loss

    def embed(self, X, adj):
        return self.encoder(X, adj).squeeze().data.cpu().numpy()
