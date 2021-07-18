# coding=utf-8
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import network
from torch.utils.data import Dataset
import networkx as nx


class EmbeddingDataset(Dataset):
    def __init__(self, graph, embedding, args):
        self.graph = graph
        self.embedding = embedding
        self.args = args

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            return idx.tolist()
        node = int(self.embedding[idx].numpy()[0])
        node_embedding = self.embedding[node].numpy()[1:]
        neighbor_embedding = self.embedding[
                                 self.graph[node][:self.args.sampling_size]].numpy()[:, 1:]
        label = np.log(len(self.graph[node]) + 1)
        # label = np.exp(self.ricci[node])
        sample = {
            'node': node,
            'node_embedding': torch.tensor(node_embedding),
            'neighbor_embedding': torch.tensor(neighbor_embedding),
            'label': label
        }
        # sample = {'node_embedding': self.embedding[idx],
        #           'neighbor_embedding': self.embedding[self.graph[idx][
        #           :self.args.sampling_size]],
        #           'label': np.log(len(self.graph[idx]) + 1)}
        return sample


class LSTMModel(nn.Module):
    def __init__(self, graph, args):
        super().__init__()
        self.graph = graph
        self.args = args
        self.degree_max = network.get_max_degree(self.graph)
        self.degree = network.get_degree(self.graph)
        self.lstm = nn.LSTM(self.args.embedding_size, self.args.embedding_size,
                            batch_first=True).cuda()
        self.mlp = nn.Linear(self.args.embedding_size, 1).cuda()

    def forward(self, x, h0, c0):
        # print(x)
        # x = x.view(len(x), -1, self.args.embedding_size).float()
        # hidden = (
        #     torch.randn(1, self.args.batch_size, self.args.embedding_size),
        #     torch.randn(1, self.args.batch_size, self.args.embedding_size))
        output, (h, c) = self.lstm(x, (h0, c0))
        lstm_out = h.view(-1, self.args.embedding_size)
        degree = F.selu(self.mlp(lstm_out))
        return lstm_out, degree
