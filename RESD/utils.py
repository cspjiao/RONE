# coding=utf-8
import logging
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from models.logreg import LogReg
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

from GraphRicciCurvature import OllivierRicci
from Walker import RootedWalker

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


def sampling_neighbors(G, node, samples=50):
    # 取一个节点的邻居，返回值的第一位为源节点
    nodeset = [node]
    neighbors = list(G[node])
    if samples >= len(neighbors):
        nodeset.extend(neighbors)
    else:
        nodeset.extend(random.sample(neighbors, samples))
    result = list(set(nodeset))
    result.sort(key=nodeset.index)
    return result


def second_similarity(G):
    n = G.number_of_nodes()
    s = np.zeros((n, n))
    neighbor = [0] * n
    for node in G.nodes:
        neighbor[node] = list(G[node])
    for i in range(n):
        for j in range(i + 1, n):
            inter = [k for k in neighbor[i] if k in neighbor[j]]
            outer = list(set(neighbor[i] + neighbor[j]))
            s[i, j] = float(len(inter)) / len(outer)
            s[j, i] = float(len(inter)) / len(outer)
    return s


def load_data(graph_path):
    graph = pd.read_csv(graph_path, header=None, sep=' ')
    G = nx.from_edgelist(graph.values.tolist())
    G.remove_edges_from(nx.selfloop_edges(G))
    # feature = pd.read_csv(feature_path, sep=',').values
    return G


def get_degree_list(G):
    return np.array(sorted(dict(G.degree).items(), key=lambda x: x[0]))[:, 1]


def random_walk_embedding(G, iteration=1000, walk_length=3):
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = 1.0

    N = len(G.nodes())
    E = len(G.edges())
    walker = RootedWalker(G, 1, 1)
    start = time.time()
    walker.preprocess_transition_probs()
    sentences = walker.simulate_walks(iteration, walk_length)
    finish = time.time()
    print(finish - start)
    print(N, E)
    pro_all = {}
    for s in sentences:
        ne = len(s) - 1
        if s[0] not in pro_all.keys():
            pro_all[s[0]] = {}
        for i in range(ne):
            if s[i] < s[i + 1]:
                e = (s[i], s[i + 1])
            else:
                e = (s[i + 1], s[i])
            if e not in pro_all[s[0]].keys():
                pro_all[s[0]][e] = 1
            else:
                pro_all[s[0]][e] += 1

    maxlen = 0
    minlen = 1e8
    prob = []
    for i in range(N):
        if len(pro_all[i].values()) > maxlen:
            maxlen = len(pro_all[i].values())
        if len(pro_all[i].values()) < minlen:
            minlen = len(pro_all[i].values())
        pro = list(pro_all[i].values())
        pro.sort(reverse=True)
        prob.append(pro)
    ee = [p + [0] * (maxlen - len(p)) for p in prob]
    print(maxlen, minlen)
    embed = np.array(ee) / maxlen
    print(embed.shape)
    return embed


def sample_equal_number_edges_non_edges(adj_mat, small_samples):
    edges = np.transpose(adj_mat.nonzero())
    num_edges = edges.shape[0]
    inverse_adj_mat = 1 - adj_mat
    non_edges = np.transpose(inverse_adj_mat.nonzero())
    num_non_edges = non_edges.shape[0]
    edges_sampled = edges[np.random.randint(num_edges, size=small_samples)]
    non_edges_sampled = non_edges[np.random.randint(num_non_edges, size=small_samples)]

    return edges_sampled, non_edges_sampled


def normalize_features_row(features):
    rowsum = np.array(features.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    features = d_mat_inv_sqrt.dot(features)
    return features


def normalize_features_col(features):
    colsum = np.array(features.sum(axis=0))
    print(colsum.shape)
    features /= colsum
    print(features.sum(axis=0))
    return features


def test_feature(G, alpha=0.8, method="OTD", verbose="INFO", num_hist=20):
    orc = OllivierRicci(G, alpha=alpha, method=method, verbose=verbose)
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()
    ricci_curvtures = list(nx.get_edge_attributes(G_orc, "ricciCurvature").values())
    min_cur = min(ricci_curvtures)
    max_cur = max(ricci_curvtures)
    N = G.number_of_nodes()
    num_hist = 20
    embed = np.zeros((N, num_hist))
    for n in G_orc.nodes():
        n_ric = [G_orc[n][nb]['ricciCurvature'] for nb in G_orc.neighbors(n)]
        embed[n] = np.histogram(n_ric, bins=num_hist, range=(min_cur, max_cur))[0]
    print(embed.shape)
    print(min_cur, max_cur)
    return embed


def compute_ricci(G, alpha=0.8, method="OTD", verbose="INFO"):
    orc = OllivierRicci(G, alpha=alpha, method=method, verbose=verbose)
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()
    node_ricci = nx.get_node_attributes(G_orc, "ricciCurvature")
    embed = np.zeros(G.number_of_nodes())
    for n in node_ricci.keys():
        embed[n] = node_ricci[n]
    temp = nx.get_edge_attributes(G_orc, 'ricciCurvature')
    N = G.number_of_nodes()
    edge_ricci = np.zeros((N, N))
    for key in temp.keys():
        edge_ricci[key] = temp[key]
    print(min(embed), max(embed))
    return embed, edge_ricci


def classification(embedding, lbl_path, split_ratio=0.7, loop=100):
    eval_dict = {
        'acc': 0.0,
        'f1-micro': 0.0,
        'f1-macro': 0.0,
    }
    label = pd.read_csv(lbl_path, header=None, sep=' ').values
    for _ in range(loop):
        labels_np = shuffle(label)
        nodes = labels_np[:, 0]
        labels = labels_np[:, 1]

        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        train_size = int(labels_np.shape[0] * split_ratio)
        features = embedding[nodes]
        train_x = features[:train_size, :]
        train_y = labels[:train_size, :]
        test_x = features[train_size:, :]
        test_y = labels[train_size:, :]
        clf = OneVsRestClassifier(
            LogisticRegression(class_weight='balanced', solver='liblinear', n_jobs=-1))
        clf.fit(train_x, train_y)
        y_pred = clf.predict_proba(test_x)
        y_pred = lb.transform(np.argmax(y_pred, 1))
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(test_y, 1)) / len(y_pred)
        eval_dict['acc'] += acc
        eval_dict['f1-micro'] += metrics.f1_score(np.argmax(test_y, 1), np.argmax(y_pred, 1),
                                                  average='micro')
        eval_dict['f1-macro'] += metrics.f1_score(np.argmax(test_y, 1), np.argmax(y_pred, 1),
                                                  average='macro')
    for key in eval_dict.keys():
        eval_dict[key] = round(1.0 * eval_dict[key] / loop, 4)
    print('split_ratio: {}'.format(split_ratio))
    print(eval_dict)
    return eval_dict


def torch_log_reg(embedding, lbl_path, split_ratio=0.7, loop=100):
    eval_dict = {
        'acc': 0.0,
        'f1-micro': 0.0,
        'f1-macro': 0.0,
    }
    embedding = torch.from_numpy(embedding).cuda()
    label = pd.read_csv(lbl_path, header=None, sep=' ').values
    for _ in range(loop):
        labels_np = shuffle(label)
        nodes = labels_np[:, 0]
        labels = labels_np[:, 1]
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)

        # labels = torch.from_numpy(labels).cuda()
        log = LogReg(embedding.shape[1], labels.shape[1])
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()
        train_size = int(labels_np.shape[0] * split_ratio)
        features = embedding[nodes]
        train_x = features[:train_size, :]
        train_y = torch.from_numpy(labels[:train_size, :]).cuda()
        train_y = torch.argmax(train_y, dim=1, keepdim=True).reshape(train_y.shape[0])

        test_x = features[train_size:, :]
        test_y = labels[train_size:, :]

        for _ in range(500):
            log.train()
            opt.zero_grad()

            logits = log(train_x)
            loss = F.cross_entropy(logits, train_y)

            loss.backward()
            opt.step()

        logits = log(test_x)
        preds = torch.argmax(logits, dim=1).cpu()
        acc = torch.sum(preds == np.argmax(test_y)).float() / test_y.shape[0]
        eval_dict['acc'] += acc.numpy()
        eval_dict['f1-micro'] += metrics.f1_score(np.argmax(test_y, 1), preds,
                                                  average='micro')
        eval_dict['f1-macro'] += metrics.f1_score(np.argmax(test_y, 1), preds,
                                                  average='macro')
        print(eval_dict)
    for key in eval_dict.keys():
        eval_dict[key] = round(1.0 * eval_dict[key] / loop, 4)
    print('split_ratio: {}'.format(split_ratio))
    print(eval_dict)
    return eval_dict


def _k_precision(embedding, lbl_path, k, lbl):
    label = pd.read_csv(lbl_path, header=None, sep=' ').values
    nodes = label[np.where(label[:, 1] == lbl)][:, 0]
    acc = 0.0
    for node in nodes:
        distance = {}
        for i in range(embedding.shape[0]):
            if i == node:
                continue
            distance[i] = np.linalg.norm(embedding[i] - embedding[node])
        distance = sorted(distance.items(), key=lambda x: x[1])
        distance = np.array(distance)[:k]
        acc += distance[np.isin(distance[:, 0], nodes)].shape[0] / k
    acc /= len(nodes)
    return acc


def k_precision(embedding, lbl_path, k=50):
    eval_dict = {
        'precision': k,
        'bots_acc': _k_precision(embedding, lbl_path, k, 1),
        'admins_acc': _k_precision(embedding, lbl_path, k, 2)
    }
    print(eval_dict)


if __name__ == '__main__':
    G = nx.read_edgelist('../dataset/clf/usa-flights.edge', nodetype=int)
    s = second_similarity(G)
    print(s.shape)
