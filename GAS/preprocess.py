# -*- coding: utf-8 -*-
from fastdtw import fastdtw
from tqdm import tqdm
import os
import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp
import torch
from time import time
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor, as_completed

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense()

def normalize_adj1(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).todense()

def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def feature_extractor(graph, node):
    # referenced ReFeX
    nbs = list(nx.neighbors(graph, node))
    sub_nodes = nbs + [node]
    sub_g = nx.subgraph(graph, sub_nodes)
    overall_counts = np.sum(list(map(lambda x: len(list(nx.neighbors(graph,x))), sub_nodes)))
    in_counts = len(nx.edges(sub_g))
    degree = nx.degree(sub_g, node)
    trans = nx.clustering(sub_g, node)
    triangles = nx.triangles(sub_g, node)
    if in_counts > 0:
        ft1 = float(in_counts)/float(overall_counts)
        ft2 = float(overall_counts - in_counts)/float(overall_counts)
    else:
        ft1 = 0
        ft2 = 0
    return [in_counts, overall_counts, ft1, ft2,degree, trans, triangles]

def branch_list(graph, node):
    branch_list = []
    nbs = list(nx.neighbors(graph, node))
    degree = nx.degree(graph, node)
    branch_list = branch_list+ [1/(degree+1) for i in range(degree)]
    degree1hop = [nx.degree(graph, node) for i in nbs]
    for d1 in degree1hop:
        branch_list = branch_list + [1/(degree+1)/(d1+1) for i in range(d1-1)]
    return branch_list

def tree_dist(a,b):
    distance, _ = fastdtw(a, b, radius=1)
    return distance

def ft_dist(a,b):
    return np.linalg.norm(a-b,ord=2)

def load_data(args):
    print("Loading Data...")
    if 'clf/' in args.dataset:
        graph = nx.from_edgelist(pd.read_csv("../dataset/" + args.dataset + ".edge", encoding='utf-8', header=None, sep=' ').values.tolist())
        graph.remove_edges_from(nx.selfloop_edges(graph))
    if 'lp/' in args.dataset:
        datafile = open("../cache/" + args.dataset + "-1.pkl",'rb')
        graphAttr = pkl.load(datafile)
        graph = graphAttr['G_rmd']
        datafile.close()
    pfile = 'pickles/'+args.dataset+'.pkl'
    nodeNum = len(graph.nodes())
    if os.path.exists(pfile):
        print('Loading Pickle File...')
        with open(pfile,'rb') as pf:
            Dist = pkl.load(pf)
    else:
        print('Calculating Node Structure Distance based on Features...')
        print('Extracting Features...')
        t0 = time()
        base_features = []
        for i in tqdm(range(nodeNum)):
            base_features.append(feature_extractor(graph, i))
        features = np.array(base_features)
        adj = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))
        adj = adj + np.eye(adj.shape[0])
        adj_norm = normalize_adj1(adj)
        features = np.concatenate((features, np.matmul(adj_norm, features),np.matmul(adj,features)), axis=1)
        features_norm = features / features.max(axis=0)
        #features_norm = np.divide(features, features.max(axis=0),out=np.zeros_like(features),where=features.max(axis=0)!=0)
        t1 = time()
        print(t1-t0)
        t0 = time()
        Dist = {'F':features_norm }
        with open(pfile,'wb') as pf:
            pkl.dump(Dist, pf)
        t1 = time()
        print(t1-t0)
    degree = sorted(graph.degree())
    D = np.zeros((nodeNum,nodeNum))
    for i in range(nodeNum):
        D[i][i]=degree[i][1]
    return graph, Dist, nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes())), D
