# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Discriminator_DGI(nn.Module):
    def __init__(self, n_h):
        super(Discriminator_DGI, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.init_weights(m)

    def init_weights(self, m):
        if isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_p1, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c,1)
        c_x = c_x.expand_as(h_p1)

        sc_1 = torch.squeeze(self.f_k(h_p1, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

