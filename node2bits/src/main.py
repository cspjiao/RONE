import sys
import datetime
from pathlib import Path
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
import pickle as pkl
import itertools
import argparse
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds, eigs
import sparsesvd
import scipy.spatial.distance as distance
from task import Task
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, DictionaryLearning
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.feature_extraction import FeatureHasher

import collections
from collections import defaultdict

from util import *
import time

def get_sketch(K, P, idx):
    file_name = 'sketch_' + str(idx) + '.tsv'
    tmp = np.random.choice([-1, 1], size=(int(K), P))
    # np.savetxt(file_name, tmp, delimiter='\t', fmt='%i')
    return tmp


def bit_to_int(binary_list):
    return int(''.join([str(int(ele)) for ele in binary_list]), 2)


def hash_func(PSs, Ks, Ts):
    print('[Hasing cosine sketch begins]')
    b = 4
    tables = [defaultdict(list) for _ in range(len(PSs))]
    f_rep = None

    for idx in range(len(PSs)):

        PS = PSs[idx]
        K = Ks[idx]
        T = Ts[idx]
        B = K / b

        # table = tables[idx]
        indices_back = [b * i for i in range(int(B))]
        indices = random.sample(indices_back,
                                int(B / 2))  # OR-construction: randomly select B-T bands
        print('[indices] ' + str(indices))

        N, P = PS.shape

        sketches = get_sketch(K, P, idx)
        # print sketches.shape
        projection = np.matmul(PS, sketches.T).astype(int)
        result = (np.sign(projection) + 1) / 2

        if idx == 0:
            f_rep = result
        else:
            f_rep = np.concatenate((f_rep, result), axis=1)

        ##################################################################

        '''for node in range(N):
            row = result[node, :]
            signature = tuple(  [ bit_to_int(row[index:index + b]) for index in indices]  ) 
            # sign format: tuple
            table[signature].append(node)'''

    return tables, f_rep


def get_init_features(graph, base_features, nodes_to_explore):
    init_feature_matrix = np.zeros((len(nodes_to_explore), len(base_features)))
    adj = graph.adj_matrix

    if "outdegree" in base_features:
        init_feature_matrix[:, 0] = (adj.sum(axis=0).transpose() + adj.sum(axis=1)).ravel()

    if "indegree" in base_features:
        init_feature_matrix[:, 1] = adj.sum(axis=0).transpose().ravel()

    if "degree" in base_features:
        init_feature_matrix[:, 2] = adj.sum(axis=1).ravel()

    return init_feature_matrix


def get_feature_n_buckets(feature_matrix, num_buckets, bucket_max_value):
    result_sum = 0
    result_ind = []
    result_cum = []
    N, cur_P = feature_matrix.shape

    if num_buckets is not None:
        for i in range(cur_P):
            result_cum.append(result_sum)
            n_buckets = max(bucket_max_value,
                            int(math.log(max(max(feature_matrix[:, i]), 1), num_buckets) + 1))
            result_sum += n_buckets
            result_ind.append(n_buckets)

    else:
        for i in range(cur_P):
            result_cum.append(result_sum)
            n_buckets = max(bucket_max_value, int(max(feature_matrix[:, i])) + 1)
            result_sum += n_buckets
            result_ind.append(n_buckets)

    return result_sum, result_ind, result_cum


def feature_binning(graph, init_feature_matrix, nodes_to_explore, S, i):
    feature_wid_sum, feature_wid_ind, feature_wid_cum = get_feature_n_buckets(
        init_feature_matrix, graph.num_buckets,
        graph.bucket_max_value)
    feature_matrix_seq = np.zeros(
        [graph.num_nodes, feature_wid_sum * len(graph.cat_dict.keys())])

    N, P = init_feature_matrix.shape
    id_cat_dict = graph.id_cat_dict
    # print('id_cat_dict',id_cat_dict)
    # print('nodes_to_explore',nodes_to_explore)
    for node in nodes_to_explore:
        if node % 500 == 0:
            print("[Generate combined feature vetor] node: " + str(node))

        S_list = S[node][i]
        cur_neighbors = set(S_list)
        cur_neighbor_dict = dict(
            [x, S_list.count(x) / float(len(S_list))] for x in set(S_list))
        # print('cur_neighbor_dict',cur_neighbor_dict)
        for neighbor in cur_neighbors:

            cat_idx = id_cat_dict[neighbor]

            for p in range(P):

                feature_idx = feature_wid_cum[p]
                node_feature = init_feature_matrix[neighbor, p]

                if (graph.num_buckets is not None) and (node_feature != 0):
                    bucket_index = max(int(math.log(node_feature, graph.num_buckets)), 0)
                else:
                    bucket_index = int(node_feature)

                bucket_index = min(bucket_index, (feature_wid_ind[p] - 1))
                global_idx = cat_idx * feature_wid_sum + feature_idx + bucket_index
                feature_matrix_seq[node, global_idx] += cur_neighbor_dict[neighbor]
                # print('feature_matrix_seq',feature_matrix_seq)

    return feature_matrix_seq


def construct_prox_structure(graph, nodes_to_explore, base_features, S_out, dist_scope):
    init_feature_matrix = get_init_features(graph, base_features, nodes_to_explore)

    feature_matrices = []
    # print(S_out)
    for i in range(dist_scope):
        feature_matrices.append(
            feature_binning(graph, init_feature_matrix, nodes_to_explore, S_out, i))
    # print('feature_matrices',feature_matrices)
    return feature_matrices


def parse_weighted_temporal(dataset, delimiter):
    check_eq = True
    num_nodes = 0
    num_edges = 0
    adj_matrix_global = None
    edge_time_dict = None
    time_edge_dict = None
    start_time = 0
    end_time = 0

    '''raw = np.genfromtxt(input_file_path, dtype=int, delimiter=delimiter)
    ROW, COL = raw.shape
    num_edges = ROW'''
    t1 = 0
    if "/clf/" in input_file_path:
        graph = nx.from_edgelist(pd.read_csv(input_file_path, encoding='utf-8', header=None,
                                             sep=' ').values.tolist())
        t1=time.time()
        graph.remove_edges_from(nx.selfloop_edges(graph))
        num_edges = len(graph.edges())
    elif "/lp/" in input_file_path:
        datafile = open(input_file_path, 'rb')
        graphAttr = pkl.load(datafile)
        graph = graphAttr['G_rmd']
        datafile.close()
        num_edges = graphAttr['edges']
    num_nodes = len(graph.nodes())

    adj_matrix_global = sps.lil_matrix(
        nx.to_scipy_sparse_matrix(graph, nodelist=list(range(num_nodes))))
    '''if COL == 3:
        print('[input_file does not contain timestamps. Processing as static graphs]')

        srcs = raw[:,0]
        dsts = raw[:,1]
        weis = raw[:,2]
        max_id = int(max(max(srcs), max(dsts)))
        num_nodes = max_id + 1
        print('[max_node_id] ' + str(max_id))
        print('[num_nodes] ' + str(num_nodes))

        if max(srcs) != max(dsts):
            srcs = np.append(srcs, max(max(srcs), max(dsts)))
            dsts = np.append(dsts, max(max(srcs), max(dsts)))
            weis = np.append(weis, 0)
            check_eq = False
        adj_matrix_global = sps.lil_matrix( sps.csc_matrix((weis, (srcs, dsts))))

    elif COL == 4:
        print('[input_file contains timestamps. Processing as dynamic graphs]')

        edge_time_dict = defaultdict(list)
        time_edge_dict = defaultdict(list)

        srcs = raw[:,0]
        dsts = raw[:,1]
        weis = raw[:,2]
        times = raw[:,3]

        start_time = min(times)
        end_time = max(times)

        max_id = int(max(max(srcs), max(dsts)))
        num_nodes = max_id + 1
        print('[max_node_id] ' + str(max_id))
        print('[num_nodes] ' + str(num_nodes))

        if max(srcs) != max(dsts):
            srcs = np.append(srcs, max(max(srcs), max(dsts)))
            dsts = np.append(dsts, max(max(srcs), max(dsts)))
            weis = np.append(weis, 0)
            check_eq = False

        adj_matrix_global = sps.lil_matrix( sps.csc_matrix((weis, (srcs, dsts))))

        fIn = open(input_file_path, 'r')
        lines = fIn.readlines()
        for line in lines:
            parts = line.strip('\r\n').split(delimiter)
            src = int(parts[0])
            dst = int(parts[1])
            wei = float(parts[2])
            timestamp = int(parts[3])

            edge = (src, dst, wei)
            edge_time_dict[edge].append(timestamp)
            time_edge_dict[timestamp].append(edge)

        fIn.close()


    else:
        sys.exit('[input_file format error. Please make sure the input file with the format 
        <src, dst, wei> or <src, dst, wei, timestamps>')'''

    return check_eq, num_nodes, num_edges, adj_matrix_global, edge_time_dict, \
           time_edge_dict, start_time, end_time, t1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def construct_cat_tmp(num_nodes):
    result = defaultdict(set)
    id_cat_dict = dict()
    for n in range(num_nodes):
        result[0].add(n)
        id_cat_dict[n] = 0
    return result, id_cat_dict


def construct_cat(input_gt_path, delimiter):
    ####################################################
    # Input: per line, 1) cat-id_init, id_end or 2) cat-id
    #
    # Return: 1) dict: cat-ids and 2) id-cat
    ####################################################

    result = defaultdict(set)
    id_cat_dict = dict()

    fIn = open(input_gt_path, 'r')
    lines = fIn.readlines()
    for line in lines:

        parts = line.strip('\r\n').split(delimiter)
        if len(parts) == 3:
            cat = parts[0]
            node_id_start = parts[1]
            node_id_end = parts[2]

            for i in range(int(node_id_start), int(node_id_end) + 1):
                result[int(cat)].add(i)
                id_cat_dict[i] = int(cat)

        elif len(parts) == 2:
            cat = parts[0]
            node_id = parts[1]

            result[int(cat)].add(int(node_id))
            id_cat_dict[int(node_id)] = int(cat)

        else:
            sys.exit('Cat file format not supported')

    fIn.close()
    return result, id_cat_dict


def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(
        description="node2bits: Compact Time- and Attribute-aware Node Representations for "
                    "User Stitching.")
    parser.add_argument('--dataset',
                        nargs='?',
                        default="clf/brazil-flights",
                        help='dataset')
    parser.add_argument('--lpmethod', nargs='?', default='Hadamard', help='binary operator')
    parser.add_argument('--bin', nargs='?', default=True, help='Binary or not')

    parser.add_argument('--input', nargs='?', default='graph/test.tsv',
                        help='Input graph file path')

    parser.add_argument('--cat', nargs='?', default='graph/test_cat.tsv',
                        help='Input node category file path')

    parser.add_argument('--output', nargs='?', default='emb/test_emb.txt',
                        help='Embedding file path')

    parser.add_argument('--dim', type=int, default=128, help='Embedding dimension')

    parser.add_argument('--scope', type=int, default=3, help='Temporal distance to consider')

    parser.add_argument('--base', type=int, default=4,
                        help='Base constant of logarithm histograms')

    parser.add_argument('--walk_num', type=int, default=10,
                        help='The number of walks per node')

    parser.add_argument('--walk_length', type=int, default=20, help='The length of the walk')

    parser.add_argument('--walk_mod', nargs='?', default='early',
                        help='The random walk mode. Could be <random>, <early> or <late>.')

    parser.add_argument('--ignore_time', type=str2bool, default=False,
                        help='Ignore the timestamps. Only used when the graph is dynamic.')

    return parser.parse_args()


if __name__ == '__main__':
    start = time.time()
    args = parse_args()

    input_file_path = args.input
    input_gt_path = args.cat
    output_file_path = args.output
    dim = args.dim
    scope = args.scope
    num_buckets = args.base
    walk_num = args.walk_num
    walk_length = args.walk_length
    walk_mod = args.walk_mod
    graph_mod_external = args.ignore_time

    print('----------------------------------')
    print('[Input graph file] ' + input_file_path)
    print('[Input category file] ' + input_gt_path)
    print('[Output embedding file] ' + output_file_path)
    print('[Embedding dimension] ' + str(dim))
    print('[Value of scope] ' + str(scope))
    print('[Base of logarithm binning] ' + str(num_buckets))
    print('[The number of walks] ' + str(walk_num))
    print('[The length of a walk] ' + str(walk_length))
    print('----------------------------------')

    ##########################################
    # Initialize
    ##########################################

    # delimiter = get_delimiter(input_file_path)
    delimiter = '\t'
    '''adj_matrix: lil format adj matrix
        edge_time_dict: (src, dst, wei) - time_1, time_2, ...
    '''
    if 'clf/' in args.dataset:
        input_file_path = "../dataset/" + args.dataset + ".edge"
    elif 'lp/' in args.dataset:
        input_file_path = "../cache/" + args.dataset + "-1.pkl"
    check_eq, num_nodes, num_edges, adj_matrix, edge_time_dict, time_edge_dict, start_time, \
    end_time, t1 = parse_weighted_temporal(
        input_file_path, delimiter)
    ##########################################
    # Setup
    ##########################################
    nodes_to_explore = range(num_nodes)
    base_features = ['degree', 'indegree', 'outdegree']

    Ks = [dim / scope] * (scope - 1)
    Ks += [dim - sum(Ks)]
    Ts = [4 for _ in range(len(Ks))]

    # The initial timestamp to perform random walk. 'late' mode would result in short walks
    # with smaller node contexts.
    init_mod = 'early'

    ##########################################

    graph_mod = 'static' if edge_time_dict is None else 'dynamic'
    print('[Graph mode detected] ' + graph_mod)

    if graph_mod is 'dynamic' and graph_mod_external is True:
        graph_mod = 'static'
        print('[Graph mode set as] ' + graph_mod)

    SM = Static_Methods(adj_matrix=adj_matrix, nodes_to_explore=nodes_to_explore)
    DM = Dynamic_Methods(adj_matrix=adj_matrix, nodes_to_explore=nodes_to_explore,
                         edge_time_dict=edge_time_dict)
    neighbor_list_static = SM.construct_neighbor_list()
    neighbor_list_dynamic = DM.construct_neighbor_list()

    # CAT_DICT, ID_CAT_DICT = construct_cat(input_gt_path, delimiter)
    CAT_DICT, ID_CAT_DICT = construct_cat_tmp(num_nodes)
    # print(CAT_DICT, ID_CAT_DICT)

    G = Graph(adj_matrix=adj_matrix, edge_time_dict=edge_time_dict, num_nodes=num_nodes,
              num_edges=num_edges,
              check_eq=check_eq,
              start_time=start_time, end_time=end_time, num_buckets=num_buckets,
              bucket_max_value=30, cat_dict=CAT_DICT,
              id_cat_dict=ID_CAT_DICT,
              neighbor_list_static=neighbor_list_static,
              neighbor_list_dynamic=neighbor_list_dynamic,
              time_edge_dict=time_edge_dict, dist_scope=scope)

    walks, S_out = G.simulate_walks(walk_num, walk_length, nodes_to_explore, walk_mod,
                                    graph_mod, init_mod)

    PSs = construct_prox_structure(G, nodes_to_explore, base_features, S_out, scope)
    tables, rep = hash_func(PSs, Ks, Ts)
    if args.bin:
        embedding = rep
    else:
        embedding = PSs
    t2 = time.time()
    print("Embedding time: {}".format(t2-t1))
    timepath = '../runningtime.txt'
    f = open(timepath, "a+")
    f.write("node2bits {} running time :{}\n".format(args.dataset, t2 - t1))
    f.close()
    print('[Write binary embeddings]')
    columns = ["id"] + ["x_" + str(x) for x in range(embedding.shape[1])]
    ids = np.array([n for n in range(num_nodes)]).reshape(-1, 1)
    embedding = pd.DataFrame(np.concatenate((ids, embedding), axis=1), columns=columns)
    embedding = embedding.sort_values(by=['id'])
    embedding.to_csv("../embed/node2bits/" + args.dataset + '_' + str(args.dim) + ".emb",
                     index=None)
    # print(embedding)
    print(num_nodes, num_edges)
    print("Total time: %s" % (time.time() - start))
