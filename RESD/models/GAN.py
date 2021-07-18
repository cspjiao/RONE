# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.generator = nn.ModuleList(
            [nn.Linear(self.config.struct[-1], self.config.struct[-1]) for i in
             range(len(self.config.struct) - 1)]).to(self.config.device, dtype=torch.float32)
        for i in range(len(self.generator)):
            nn.init.xavier_uniform_(self.generator[i].weight)
            nn.init.uniform_(self.generator[i].bias)

    def forward(self, h_state):
        for i in range(len(self.config.struct) - 1):
            h_state = torch.tanh(self.generator[i](h_state))
        return h_state


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.discriminator = nn.ModuleList([
            nn.Linear(self.config.struct[-1], self.config.struct[-1]),
            nn.Linear(self.config.struct[-1], self.config.struct[-1]),
            nn.Linear(self.config.struct[-1], 1),
        ]).to(
            self.config.device,
            dtype=torch.float32)
        for i in range(len(self.discriminator)):
            nn.init.xavier_uniform_(self.discriminator[i].weight)
            nn.init.uniform_(self.discriminator[i].bias)

    def forward(self, h_state):
        for i, layer in enumerate(self.discriminator):
            h_state = layer(h_state)
            if i != len(self.discriminator) - 1:
                h_state = F.relu(h_state)
        return h_state
