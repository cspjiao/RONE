# coding=utf-8
import argparse
import numpy as np
import time
import network
import testmodel
from task import Task
import torch
import torch.nn as nn
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import torch.optim as optim
import os
import pandas as pd
from utils1 import _classification
import warnings
from utils import compute_ricci
import networkx as nx
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        "Deep Recursive Network Embedding with Regular Equivalence")
    parser.add_argument('--data_path', type=str, default="barbell",
                        help='Directory to load data.')
    # parser.add_argument('--save_path', type=str, default="../../embed/DRNE/clf/barbell",
    # help='Directory to save data.')
    parser.add_argument('--save_suffix', type=str, default='emb',
                        help='Directory to save data.')
    parser.add_argument('-s', '--embedding_size', type=int, default=16,
                        help='the embedding dimension size')
    parser.add_argument('-e', '--epochs_to_train', type=int, default=20,
                        help='Number of epoch to train. Each epoch processes the training '
                             'data once completely')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Number of training examples processed per step')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0025,
                        help='initial learning rate')
    parser.add_argument('--undirected', type=bool, default=True,
                        help='whether it is an undirected graph')
    parser.add_argument('-a', '--alpha', type=float, default=0.001,
                        help='the rate of structure loss and orth loss')
    parser.add_argument('-l', '--lamb', type=float, default=0.5,
                        help='the rate of structure loss and guilded loss')
    parser.add_argument('-g', '--grad_clip', type=float, default=5.0, help='clip gradients')
    parser.add_argument('-K', type=int, default=1, help='K-neighborhood')
    parser.add_argument('--sampling_size', type=int, default=100, help='sample number')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--index_from_0', type=bool, default=True,
                        help='whether the node index is from zero')
    parser.add_argument('--loop', type=int, default=40,
                        help='loop')
    parser.add_argument('--gate', type=float, default=1.0,
                        help='gate')
    return parser.parse_args()


def collate(batch):
    # print(len(batch))
    batch.sort(key=lambda x: len(x['neighbor_embedding']), reverse=True)
    data_length = []
    node_embedding = []
    edge_embedding = []
    label = []
    node = []
    for data in batch:
        node.append(data['node'])
        node_embedding.append(data['node_embedding'].numpy())
        edge_embedding.append(data['neighbor_embedding'])
        label.append(data['label'])
        data_length.append(len(data['neighbor_embedding']))
    edge_embedding = pad_sequence(edge_embedding, batch_first=True)
    return {
        'node': node,
        'node_embedding': torch.FloatTensor(node_embedding),
        'neighbor_embedding': edge_embedding,
        'label': torch.FloatTensor(label),
        'data_length': data_length
    }


def main(args):
    file_path = '../dataset/clf/'
    device = 'cuda:0'
    np.random.seed(int(time.time()) if args.seed == -1 else args.seed)
    graph = network.read_from_edgelist(file_path + args.data_path + '.edge',
                                       index_from_zero=args.index_from_0)
    # G = nx.read_edgelist(file_path + args.data_path + '.edge', nodetype=int)
    # G.remove_edges_from(nx.selfloop_edges(G))
    # ricci = compute_ricci(G, alpha=0.5, method='ATD')
    network.sort_graph_by_degree(graph)
    embedding = np.vstack(
        [np.random.normal(i * 1.0 / network.get_max_degree(graph), 0.001, args.embedding_size)
         for i in
         network.get_degree(graph)])
    # print(embedding)
    ids = np.array([i for i in range(len(graph))]).reshape(-1, 1)
    embedding = np.c_[ids, embedding]
    embedding = torch.from_numpy(embedding)

    dataset = testmodel.EmbeddingDataset(graph, embedding, args)
    # print(dataset[0])
    # print(dataset.graph)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                            collate_fn=collate)
    # for i, data in enumerate(dataloader):
    #     print(i, data)
    # return
    lstm = testmodel.LSTMModel(graph, args).to(device, dtype=torch.float32)
    optimiser = optim.RMSprop(lstm.parameters(), lr=args.learning_rate)

    square_mean = nn.MSELoss()
    square_mean2 = nn.MSELoss()
    reduce_mean = nn.L1Loss()
    print('training')
    print("batch_size: {}".format(args.batch_size))
    for epoch in range(0, args.epochs_to_train):
        for i, data in enumerate(dataloader):
            node_embedding = data['node_embedding'].to(device, dtype=torch.float32)
            # print(node_embedding)
            edge_embedding = data['neighbor_embedding'].to(device, dtype=torch.float32)
            data_length = data['data_length']
            edge_embedding = pack_padded_sequence(edge_embedding, data_length,
                                                  batch_first=True)
            # print(edge_embedding)
            label = torch.FloatTensor(data['label']).view(-1, 1).to(device,
                                                                    dtype=torch.float32)
            optimiser.zero_grad()
            # print(node_embedding.size())
            h0 = torch.zeros(1, node_embedding.size()[0], args.embedding_size).to(device,
                                                                                  dtype=torch.float32)
            c0 = torch.zeros(1, node_embedding.size()[0], args.embedding_size).to(device,
                                                                                  dtype=torch.float32)
            lstm_out, mlp_out = lstm(edge_embedding.to(device, dtype=torch.float32), h0, c0)
            # print(lstm_out.size())
            # print(mlp_out.size())
            # print(lstm_out.size())
            structure_loss = square_mean(node_embedding, lstm_out)
            guilded_loss = reduce_mean(label, mlp_out)
            orth_loss = square_mean2(lstm_out.t().mm(lstm_out),
                                     torch.ones(args.embedding_size,
                                                args.embedding_size).to(device,
                                                                        dtype=torch.float32))
            total_loss = args.gate * structure_loss + args.lamb * guilded_loss + args.alpha * \
                         orth_loss

            total_loss.backward()
            optimiser.step()
        print((
            "epoch: {}/{}, loss: {:.6f}, structure_loss: {:.6f}, orth_loss: {:.6f}, "
            "guilded_loss: {:.6f},").format(
            epoch, args.epochs_to_train, total_loss, structure_loss, orth_loss,
            guilded_loss))
    out_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    out_embedding = []
    with torch.no_grad():
        for data in out_dataloader:
            node_embedding = data['node_embedding']
            # print(node_embedding)
            edge_embedding = data['neighbor_embedding'].to(device, dtype=torch.float32)
            h0 = torch.zeros(1, node_embedding.size()[0], args.embedding_size).to(
                device, dtype=torch.float32)
            c0 = torch.zeros(1, node_embedding.size()[0], args.embedding_size).to(
                device, dtype=torch.float32)
            lstm_out, mlp_out = lstm(edge_embedding, h0, c0)
            out_embedding.append(lstm_out.view(args.embedding_size).cpu().numpy().tolist())

        # print(out_embedding)
        # lbl_path = '../dataset/clf/{}.lbl'.format(args.data_path)
        # out_embedding = np.array(out_embedding)
        # eval_dict = _classification(out_embedding, lbl_path, split_ratio=0.7,
        #                             loop=args.loop,
        #                             seed=args.seed)
    # save_embedding
    task = Task('CLF')
    label = np.loadtxt("../dataset/clf/" + args.data_path + '.lbl', dtype=np.int)
    task.classfication(np.array(embedding), label, split_ratio=0.7, loop=20)
    save_path = '../embed/DRNE/clf'
    file_name = '%s.emb' % args.data_path.replace('.edge', '')
    print("Save embeddings in {}/{}".format(save_path, file_name))
    columns = ["id"] + ["x_" + str(x) for x in range(args.embedding_size)]
    out_embedding = pd.DataFrame(np.concatenate([ids, out_embedding], axis=1), columns=columns)
    out_embedding = out_embedding.sort_values(by=['id'])
    out_embedding.to_csv(os.path.join(save_path, file_name), index=None)


if __name__ == '__main__':
    start = time.time()
    main(parse_args())
    end = time.time()
    print('Embeddings time:{}s'.format(end - start))
