# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, G, config):
        super(VAE, self).__init__()
        print(config)
        self.N = G.number_of_nodes()
        self.config = config
        self.encoder = nn.ModuleList(
            [nn.Linear(self.config.struct[i], self.config.struct[i + 1]) for i in
             range(len(self.config.struct) - 1)]).to(self.config.device, dtype=torch.float32)
        self.enc_mu = nn.Linear(self.config.struct[-1], self.config.struct[-1]).to(
            self.config.device, dtype=torch.float32)
        self.enc_log_sigma = nn.Linear(self.config.struct[-1], self.config.struct[-1]).to(
            self.config.device, dtype=torch.float32)
        # self.config.struct[0] = self.N
        self.config.struct.reverse()
        self.decoder = nn.ModuleList(
            [nn.Linear(self.config.struct[i], self.config.struct[i + 1]) for i in
             range(len(self.config.struct) - 1)]).to(self.config.device, dtype=torch.float32)
        self.config.struct.reverse()

        # self.encoder = GCNEncoder(G, config).to(self.config.device, dtype=torch.float32)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.init_model_weight()

    def init_model_weight(self):
        for i in range(len(self.config.struct) - 1):
            nn.init.xavier_uniform_(self.encoder[i].weight)
            nn.init.uniform_(self.encoder[i].bias)
        for i in range(len(self.config.struct) - 1):
            nn.init.xavier_uniform_(self.decoder[i].weight)
            nn.init.uniform_(self.decoder[i].bias)

    def encoder_network(self, h_state):
        for i in range(len(self.config.struct) - 1):
            h_state = F.tanh(self.encoder[i](h_state))
        mu = self.enc_mu(h_state)
        log_sigma = self.enc_log_sigma(h_state)
        sigma = log_sigma.exp()
        z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).to(
            self.config.device, dtype=torch.float32)

        z = mu + sigma * z

        return mu, sigma, z

    def decoder_network(self, h_state):
        for i, layer in enumerate(self.decoder):
            # if i != len(self.decoder) - 1:
            #     h_state = torch.tanh(layer(h_state))
            # else:
            #     h_state = layer(h_state)
            h_state = layer(h_state)
            if i != len(self.decoder) - 1:
                h_state = F.tanh(h_state)
        return h_state

    def forward(self, h_state):
        mu, sigma, z = self.encoder_network(h_state)
        return mu, sigma, z, self.decoder_network(z)

    def get_embedding(self, h_state):
        mu, sigma, z = self.encoder_network(h_state)
        return z


# class VGAE(nn.Module):
#     def __init__(self, G, config):
#         super(VGAE, self).__init__()
#         print(config)
#         self.config = config
#         self.config.struct.reverse()
#         self.decoder = nn.ModuleList(
#             [nn.Linear(self.config.struct[i], self.config.struct[i + 1]) for i in
#              range(len(self.config.struct) - 1)]).to(self.config.device, dtype=torch.float32)
#         self.config.struct.reverse()
#         self.encoder = GCNEncoder(G, config).to(self.config.device, dtype=torch.float32)
#
#     def decoder_network(self, h_state):
#         for i in range(len(self.config.struct) - 1):
#             h_state = self.decoder[i](h_state)
#             if i != len(self.decoder) - 1:
#                 h_state = F.tanh(h_state)
#         return h_state
#
#     def forward(self, h_state):
#         mu, sigma, z = self.encoder(h_state)
#         return mu, sigma, z, self.decoder_network(z)
#
#     def get_embedding(self, h_state):
#         _, _, z = self.encoder(h_state)
#         return z
