#-*- coding: utf-8 -*-
from torch.utils import data
import pickle
import os


class roleData(data.Dataset):

    def __init__(self, graph):
        self.nodes = list(graph.nodes)

    def __getitem__(self, index):
        return self.nodes[index]

    def __len__(self):
        return len(self.nodes)
