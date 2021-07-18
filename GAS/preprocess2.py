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
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense()

def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def load_data1(args):
    print("Loading Data")
    if 'clf/' in args.dataset:
        graph = nx.from_edgelist(pd.read_csv("../dataset/" + args.dataset + ".edge", encoding='utf-8', header=-1, sep=' ').values.tolist())
        graph.remove_edges_from(graph.selfloop_edges())
    elif 'lp/' in args.dataset:
        datafile = open("../cache/" + args.dataset + "-1.pkl",'rb')
        graphAttr = pkl.load(datafile)
        graph = graphAttr['G_rmd']
        datafile.close()
    adj = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))
    if args.attr == 'deg':
        degreeList = [nd[1] for nd in sorted(nx.degree(graph, graph.nodes()))]
        degreeSet = sorted(list(set(degreeList)))
        degreeNum = len(degreeSet)
        degreeIndex = [degreeSet.index(d) for d in degreeList]
        ft = np.eye(degreeNum)[degreeIndex]
    elif args.attr == 'adj':
        ft = adj + sp.eye(adj.shape[0])
    elif args.attr == 'rolx':
        ft = pd.read_csv("../embed/ReFeX/"+args.dataset+'.emb', encoding='utf-8', sep=',').drop(['id'],axis=1).values
    elif args.attr == 'deg2sort':
        degree_1st = adj.sum(axis=0)
        tmp = np.concatenate((degree_1st, degree_1st),axis=0)
        for i in range(2,adj.shape[0]):
            tmp = np.concatenate((tmp,degree_1st),axis=0)
        tmp = torch.FloatTensor(tmp).to(device=args.device, dtype=torch.float32)
        adj_t = torch.FloatTensor(adj).to(device=args.device, dtype=torch.float32)
        tmp = tmp.mul(adj_t).cpu().numpy()
        degree_2nd = -np.sort(-tmp)
        dim = int(np.max(degree_1st))
        print(dim)
        degree_2nd = degree_2nd[...,0:dim]
        ft = np.concatenate((degree_1st.T,degree_2nd),axis=1)
    elif args.attr == 'deg2nd':
        degree_1st = adj.sum(axis=0)
        tmp = np.concatenate((degree_1st, degree_1st),axis=0)
        for i in range(2,adj.shape[0]):
            tmp = np.concatenate((tmp,degree_1st),axis=0)
        tmp = torch.FloatTensor(tmp).to(device=args.device, dtype=torch.float32)
        print(tmp)
        adj_t = torch.FloatTensor(adj).to(device=args.device, dtype=torch.float32)
        degree_2nd = tmp.mul(adj_t).cpu().numpy()
        ft = np.concatenate((degree_1st.T,degree_2nd),axis=1)
    adj_sl = adj + sp.eye(adj.shape[0])
    return adj_sl,ft

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
        graph = nx.from_edgelist(pd.read_csv("../dataset/" + args.dataset + ".edge", encoding='utf-8', header=-1, sep=' ').values.tolist())
        graph.remove_edges_from(graph.selfloop_edges())
    elif 'lp/' in args.dataset:
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
        ''' t0 = time()
        base_features = []
        for i in tqdm(range(nodeNum)):
            base_features.append(feature_extractor(graph, i))
        features = np.array(base_features)
        adj = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))
        adj = adj + np.eye(adj.shape[0])
        adj_norm = normalize_adj1(adj)
        features = np.concatenate((features, np.matmul(adj_norm, features),np.matmul(adj,features)), axis=1)
        features_norm = features / features.max(axis=0)
        print('Calculating D_F...')
        D_F = []
        for i in range(0,features_norm.shape[0]):
            d = np.linalg.norm(features_norm[i]-features_norm,ord=2,axis=1)
            D_F.append(d.reshape(features_norm.shape[0],1))
        D_F = np.concatenate(D_F, axis = 1)
        t1 = time()
        print(t1-t0)
        t0 = time()
        print('Calculating Node Structure Distance based on Distribition Tree...')
        print('Listing the branches...')
        branches = {}
        with ProcessPoolExecutor(max_workers=10) as executor:
            for i in tqdm(range(nodeNum)):
                job = executor.submit(branch_list, graph, i)
                branches[i] = job.result()
        print('Calculating D_T...')
        D_T = np.zeros((nodeNum,nodeNum))
        with ProcessPoolExecutor(max_workers=10) as executor:
            for i in tqdm(range(nodeNum)):
                for j in range(i):
                    job = executor.submit(tree_dist,branches[i], branches[j])
                    D_T[i][j] = job.result()
                    D_T[j][i] = job.result()
        t1 = time()
        print(t1-t0)'''
        t0 = time()
        Refex = pd.read_csv("../embed/ReFeX/"+args.dataset+'.emb', encoding='utf-8', sep=',').drop(['id'],axis=1).values
        D_R = np.matmul(Refex, Refex.T)
        #D_R = D_R / D_R[0][0]
        print(time()-t0)
        t0 = time()
        Dist = {'D_R':D_R}
        with open(pfile,'wb') as pf:
            pkl.dump(Dist, pf)
        t1 = time()
        print(t1-t0)

    return graph, Dist, nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))
