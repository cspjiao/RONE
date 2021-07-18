# coding=utf-8
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .GAN import Discriminator, Generator
from .VAE import VAE
from utils import *


class Test2(nn.Module):
    # 对抗里去掉R的部分
    def __init__(self, config, graph, features):
        super(Test2, self).__init__()
        self.config = config
        self.G = graph

        # self.adj = torch.from_numpy(nx.adjacency_matrix(self.G).todense()).to(
        #     self.config.device, dtype=torch.float32)
        # self.features = self.adj + torch.eye(self.G.number_of_nodes()).to(
        #     self.config.device, dtype=torch.float32)

        self.features = torch.from_numpy(features).to(self.config.device, dtype=torch.float32)

        self.config.struct[0] = self.features.shape[1]
        self.degree = torch.from_numpy(get_degree_list(self.G)).to(self.config.device,
                                                                   dtype=torch.float32).reshape(
            -1, 1)
        self.degree = torch.log(self.degree + 1)
        # self.ricci, self.edge_ricci = compute_ricci(self.G, alpha=0.5, method="OTD")
        # self.ricci = torch.from_numpy(self.ricci).exp().to(
        #     self.config.device, dtype=torch.float32).reshape(-1, 1)
        # self.edge_ricci = torch.from_numpy(self.edge_ricci).exp().to(
        #     self.config.device, dtype=torch.float32)
        self.vae = VAE(self.G, self.config).to(self.config.device, dtype=torch.float32)
        # self.generator = Generator(self.config).to(self.config.device, dtype=torch.float32)
        self.mlp = nn.ModuleList([
            nn.Linear(self.config.struct[-1], self.config.struct[-1]),
            nn.Linear(self.config.struct[-1], 1)
        ]).to(self.config.device, dtype=torch.float32)
        for i in range(len(self.mlp)):
            nn.init.xavier_uniform_(self.mlp[i].weight)
            nn.init.uniform_(self.mlp[i].bias)
        self.mseLoss = nn.MSELoss()
        self.bceLoss = nn.BCEWithLogitsLoss()
        # self.maeLoss = nn.L1Loss()

    def generate_fake(self, h_state):
        z = torch.from_numpy(np.random.normal(0, 1, size=h_state.size())).to(
            self.config.device, dtype=torch.float32)
        return z

    # node-level loss
    def gan_loss(self, embedding):
        valid = torch.ones(embedding.size(0), 1).to(self.config.device,
                                                    dtype=torch.float32)
        fake = torch.zeros(embedding.size(0), 1).to(self.config.device,
                                                    dtype=torch.float32)
        z = self.generate_fake(embedding)
        d_logits = self.discriminator(embedding)
        real_loss = F.binary_cross_entropy_with_logits(self.discriminator(z), valid)
        fake_loss = F.binary_cross_entropy_with_logits(d_logits, fake)
        g_loss = F.binary_cross_entropy_with_logits(d_logits, valid)
        return fake_loss + real_loss + g_loss

    def mlp_out(self, embedding):
        for i, layer in enumerate(self.mlp):
            embedding = torch.relu(layer(embedding))
        return embedding

    def latent_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def forward(self, input_):
        features = self.features[input_]
        mu, sigma, embedding, vae_out = self.vae(features)

        # othr_loss = self.latent_loss(mu, sigma)

        vae_loss = self.config.alpha * F.mse_loss(vae_out, features)

        guide_loss = self.config.gamma * F.l1_loss(self.mlp_out(embedding),
                                                   self.degree[input_])

        # node_loss = self.config.beta * self.gan_loss(embedding)

        # valid = torch.ones(embedding.size(0), 1).to(self.config.device,
        #                                             dtype=torch.float32)
        # fake = torch.zeros(embedding.size(0), 1).to(self.config.device,
        #                                             dtype=torch.float32)
        # z = self.generate_fake(embedding)
        # g_loss = F.binary_cross_entropy_with_logits(self.discriminator(embedding), valid)
        #
        # fake_loss = F.binary_cross_entropy_with_logits(self.discriminator(z),
        #                                                fake)
        # real_loss = F.binary_cross_entropy_with_logits(self.discriminator(
        # embedding.detach()),
        #                                                valid)

        # print("VAE reconstruct loss: {}, Discriminator loss: {}".format(
        #     vae_loss.item(), node_loss.item()))

        return vae_loss + guide_loss

    def get_embedding(self):
        return self.vae.get_embedding(self.features)
