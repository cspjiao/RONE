# coding=utf-8
import random
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
from sklearn.utils import shuffle


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


def load_data(graph_path, feature_path):
    graph = pd.read_csv(graph_path, header=None, sep=' ')
    G = nx.from_edgelist(graph.values.tolist())
    feature = pd.read_csv(feature_path, sep=',')
    return G, feature.values


def _classification(embedding, lbl_path, split_ratio=0.7, loop=100, seed=1):
    eval_dict = {
        'acc': 0.0,
        'f1-micro': 0.0,
        'f1-macro': 0.0
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
