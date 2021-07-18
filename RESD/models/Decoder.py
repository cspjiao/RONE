# coding=utf-8
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.decoder_1 = nn.Linear(self.config.struct[0], self.config.struct[0]).to(
            self.config.device, dtype=torch.float32)
        self.decoder = nn.ModuleList(
            [nn.Linear(self.config.struct[i], self.config.struct[i + 1]) for i in
             range(len(self.config.struct) - 1)]).to(self.config.device, dtype=torch.float32)
        self.sigmod = nn.Sigmoid()
        self.init_model_weight()

    def init_model_weight(self):
        nn.init.xavier_uniform_(self.decoder_1.weight)
        nn.init.uniform_(self.decoder_1.bias, a=-0.5, b=0.5)
        for i in range(len(self.config.struct) - 1):
            nn.init.xavier_uniform_(self.decoder[i].weight)
            nn.init.uniform_(self.decoder[i].bias, a=-0.5, b=0.5)

    def forward(self, embedding1, embedding2):
        x = torch.tanh(self.decoder_1(embedding1 * embedding2))
        for i in range(len(self.config.struct) - 1):
            x = torch.tanh(self.decoder[i](x))
        return x
