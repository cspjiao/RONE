# -*- coding: utf-8 -*-
import argparse
import networkx as nx
import numpy as np
import pandas as pd
import graphwave
from graphwave.graphwave import *
import pickle as pkl
import random
from task import Task
import time


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GW.")

    parser.add_argument('--dataset',
                        nargs='?',
                        default="clf/brazil-flights",
                        help='Input graph path -- edge list csv.')
    parser.add_argument('--lpmethod', nargs='?', default='Hadamard', help='binary operator')

    return parser.parse_args()


def load_graph(graph_path):
    print(graph_path)
    if "clf/" in graph_path:
        graph = nx.from_edgelist(
            pd.read_csv(graph_path, encoding='utf-8', header=None, sep=' ').values.tolist())
        graph.remove_edges_from(nx.selfloop_edges(graph))
    elif "lp/" in graph_path:
        datafile = open(graph_path, 'rb')
        graphAttr = pkl.load(datafile)
        graph = graphAttr['G_rmd']
        datafile.close()
    return graph


def main(args):
    if 'clf/' in args.dataset:
        G = load_graph("../dataset/" + args.dataset + ".edge")
    elif 'lp/' in args.dataset:
        G = load_graph("../cache/" + args.dataset + "-1.pkl")
    print('Graph {} Loaded'.format(args.dataset))
    print(len(G.nodes()), len(G.edges()))
    start = time.time()
    chi, heat_print, taus = graphwave_alg(G, np.linspace(0, 100, 25), taus=range(19, 21),
                                          verbose=True)
    Totaltime=time.time()-start
    print("Total time: %s" % Totaltime)
    timepath = '../runningtime.txt'
    f = open(timepath, "a+")
    f.write("GraphWave {} running time :{}\n".format(args.dataset, Totaltime))
    f.close()
    columns = ["id"] + ["x_" + str(x) for x in range(chi.shape[1])]
    ids = np.array([node for node in G.nodes()]).reshape(-1, 1)
    embedding = pd.DataFrame(np.concatenate([ids, chi], axis=1), columns=columns)
    embedding = embedding.sort_values(by=['id'])
    #embedding.to_csv("../embed/GraphWave/{}_{}.emb".format(args.dataset, embedding.shape[1] - 1), index=None)
    embedding.to_csv("../embed/GraphWave/{}_64.emb".format(args.dataset), index=None)



if __name__ == "__main__":
    args = parameter_parser()
    main(args)
