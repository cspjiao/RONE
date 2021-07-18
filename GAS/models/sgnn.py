# -*- coding: utf-8 -*-
import networkx as nx
import torch
import torch.nn as nn
from preprocess import normalize_adj
from functools import reduce
import numpy as np

class SGNN(nn.Module):
    def __init__(self, G, args, X , D, device):
        super(SGNN, self).__init__()
        self.args = args
        self.G = G
        self.X = X
        self.f1 = nn.Linear(self.X.shape[0], self.args.n_h1, bias=False)
        self.f2 = nn.Linear(self.args.n_h1, self.args.n_h2, bias=False)
        self.act = self.activation(self.args.act)
        self.D = D
        self.device = device
        for m in self.modules():
            self.weights_init(m)

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

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def encoder(self, nodes):
        rep = {}
        tmprep1 = {}
        tmprep2 = {}
        for node in nodes:
            nbs1 = list(nx.neighbors(self.G, node)) +[node]
            nbs2 = {i:list(nx.neighbors(self.G, i))+[i] for i in nbs1}
            for i in nbs2.keys():
                for j in nbs2[i]:
                    if j not in tmprep1.keys():
                        tmprep1[j] = self.act(self.f1(self.X[j]))
                tmprep2[i] = self.act(self.f2(reduce(lambda x,y:x+y, [tmprep1[k] for k in nbs2[i]])))
            rep[node] = reduce(lambda x,y:x+y, [tmprep2[j] for j in nbs1])
        return rep

    def cal_loss(self, rep, nodes):
        L = torch.FloatTensor([0.0]).to(device=self.device, dtype=torch.float32)
        for i in nodes:
            for j in nodes:
                #print(torch.matmul(rep[i].unsqueeze(0), rep[j].unsqueeze(0).t()).squeeze(),self.D[i][j])
                L += torch.abs(torch.matmul(rep[i].unsqueeze(0), rep[j].unsqueeze(0).t()).squeeze() - self.D[i][j])
        return L

    def forward(self, data):
        nodes = data
        nodes = nodes.cpu().numpy()
        rep = self.encoder(nodes)
        L = self.cal_loss(rep, nodes)
        return L

    def embed(self):
        nodes = list(range(self.X.shape[0]))
        rep = self.encoder(nodes)
        rep = np.array([rep[i].data.cpu().numpy() for i in nodes])
        return rep
