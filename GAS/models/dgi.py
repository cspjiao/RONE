# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator_DGI

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation,args):
        super(DGI, self).__init__()
        self.args = args
        self.gcn = GCN(n_in, n_h, args.n, activation)
        self.gcn1 = GCN(n_in, n_h+200, args.n, activation)
        self.gcn2 = GCN(n_h+200, n_h+100,  args.n,activation)
        self.gcn3 = GCN(n_h+100, n_h,  args.n,activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator_DGI(n_h)
    def encoder(self, seq, adj):
        if self.args.l == 2:
            h1 = self.gcn1(seq, adj)
            h2 = self.gcn2(h1, adj)
            h = self.gcn3(h2, adj)
        elif self.args.l == 1:
            h = self.gcn(seq, adj)
        return h

    def forward(self, seq1, seq2, adj, samp_bias1, samp_bias2):
        h_1 = self.encoder(seq1, adj)

        c = self.read(h_1)
        c = self.sigm(c)

        #h_2 = self.encoder(seq1, seq2)
        h_2 = self.encoder(seq2, adj)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret


    def embed(self, seq, adj):
        h_1 = self.encoder(seq, adj)
        #c = self.read(h_1)

        return h_1


