#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import argparse, logging
import numpy as np
import struc2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
import pickle as pkl
import graph
from task import Task
import random
import os

logging.basicConfig(filename='struc2vec.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(message)s')


def parse_args():
    '''
    Parses the struc2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run struc2vec.")

    parser.add_argument('--dataset', nargs='?', default='clf/brazil-flights',
                        help='Input graph path')
    parser.add_argument('--lpmethod', nargs='?', default='Hadamard', help='binary operator')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--until-layer', type=int, default=4,
                        help='Calculation until the layer.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--OPT1', default=False, type=bool,
                        help='optimization 1')
    parser.add_argument('--OPT2', default=False, type=bool,
                        help='optimization 2')
    parser.add_argument('--OPT3', default=False, type=bool,
                        help='optimization 3')
    return parser.parse_args()


def read_graph():
    '''
    Reads the input network.
    '''
    logging.info(" - Loading graph...")
    if 'clf/' in args.dataset:
        G = graph.load_edgelist("../dataset/" + args.dataset + ".edge", undirected=True)
    elif 'lp/' in args.dataset:
        datafile = open("../cache/" + args.dataset + "-1.pkl", 'rb')
        graphAttr = pkl.load(datafile)
        G_tmp = graphAttr['G_rmd']
        G = graph.from_networkx(G_tmp, undirected=True)
    logging.info(" - Graph {} loaded.".format(args.dataset))
    return G


def learn_embeddings(nn):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    if not os.path.exists('../embed/struc2vec/'):
        os.makedirs('../embed/struc2vec/')
    logging.info("Initializing creation of the representations...")
    walks = LineSentence('random_walks.txt')
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1,
                     sg=1,
                     workers=args.workers, iter=args.iter)
    embedding = model.wv[str(0)][np.newaxis, :]
    for i in range(1, nn):
        embedding = np.concatenate((embedding, model.wv[str(i)][np.newaxis, :]), axis=0)
    t2 = time.time()
    columns = ["id"] + ["x_" + str(x) for x in range(embedding.shape[1])]
    ids = np.array([n for n in range(nn)]).reshape(-1, 1)
    embedding = pd.DataFrame(np.concatenate((ids, embedding), axis=1), columns=columns)
    embedding = embedding.sort_values(by=['id'])
    embedding.to_csv(
        "../embed/struc2vec/" + args.dataset + '_' + str(args.dimensions) + ".emb",
        index=None)
    logging.info("Representations created.")
    print(nn)

    return t2


def exec_struc2vec(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    if (args.OPT3):
        until_layer = args.until_layer
    else:
        until_layer = None

    G = read_graph()
    numnodes = len(G.nodes())
    t1 = time.time()
    print("Start time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))))
    G = struc2vec.Graph(G, args.directed, args.workers, untilLayer=until_layer)
    if (args.OPT1):
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs()

    if (args.OPT2):
        G.create_vectors()
        G.calc_distances(compactDegree=args.OPT1)
    else:
        G.calc_distances_all_vertices(compactDegree=args.OPT1)

    G.create_distances_network()
    G.preprocess_parameters_random_walk()
    G.simulate_walks(args.num_walks, args.walk_length)
    t2=learn_embeddings(numnodes)
    print("Embedding time: {}".format(t2 - t1))
    timepath = '../runningtime.txt'
    f = open(timepath, "a+")
    f.write("struc2vec {} running time :{}\n".format(args.dataset, t2-t1))
    f.close()
    return G, numnodes



def main(args):
    exec_struc2vec(args)

    # learn_embeddings(nn)


if __name__ == "__main__":
    args = parse_args()
    main(args)
