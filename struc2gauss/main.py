import time

import calSimRank, GenPosNegPairsbyRandom
import networkx as nx
import numpy as np
from word2gauss.embeddings import GaussianEmbedding
import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='brazil-flights', help="")
    parser.add_argument('--dimension', type=int, default=128, help="")
    return parser.parse_args()


def test_train_batch_EL_spherical(train_data, n, dim):
    # print 'length of train data: ' + str(len(training_data))

    embed = GaussianEmbedding(n, dim,
                              covariance_type='spherical',
                              energy_type='IP',
                              mu_max=2.0, sigma_min=0.5, sigma_max=1.0, eta=0.1, Closs=1.0
                              )

    for k in xrange(0, len(training_data), 100):
        embed.train_batch(training_data[k:(k + 100)])

    return embed


def get_SimRank_similarity_matrix(numOfNodes):
    simDict = calSimRank.simrank(G)
    similarity = [[0.0 for i in range(numOfNodes)] for j in range(numOfNodes)]
    for key, val in simDict.items():
        for k, v in val.items():
            similarity[key][k] = v

    return similarity


def test_train_batch_EL_diagonal(train_data, n, dim):
    # print 'length of train data: ' + str(len(training_data))

    embed = GaussianEmbedding(n, dim,
                              covariance_type='diagonal',
                              energy_type='IP',
                              mu_max=2.0, sigma_min=0.5, sigma_max=1.0, eta=0.1, Closs=1.0
                              )

    for k in xrange(0, len(train_data), 100):
        embed.train_batch(train_data[k:(k + 100)])

    return embed


def get_RoleSim_similarity_matrix(numOfNodes, infile):
    similarity = [[0.0 for i in range(numOfNodes)] for j in range(numOfNodes)]
    fin = open(infile, 'r')
    idx = 0
    for line in fin.readlines():
        tmp = line.strip().split(',')
        for i in range(numOfNodes):
            similarity[idx][i] = float(tmp[i])
        idx += 1
    fin.close()
    return similarity


if __name__ == '__main__':
    args = parse_args()
    G = nx.read_edgelist("../dataset/clf/{}.edge".format(args.dataset))
    t1 = time.time()
    print("Start time: {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))))
    numOfNodes = G.number_of_nodes()
    print(numOfNodes)
    if not os.path.exists('cache/{}.sim'.format(args.dataset)):
        os.system('RoleSim/main.o {} ../dataset/clf/{}.edge cache/{}.sim'.
                  format(numOfNodes, args.dataset, args.dataset))
    similarity = get_RoleSim_similarity_matrix(numOfNodes,
                                               'cache/{}.sim'.format(args.dataset))
    training_data = np.asarray(GenPosNegPairsbyRandom.genPosNegPairs(similarity, 40, 10),
                               dtype=np.uint32)

    embed = test_train_batch_EL_diagonal(training_data, numOfNodes, args.dimension)

    embedding = embed.mu
    covar = embed.sigma

    print(embedding.shape)
    np.savetxt('emb/{}.emb'.format(args.dataset), embedding)
    np.savetxt('emb/{}.var'.format(args.dataset), covar)
    embedding = embed.mu[numOfNodes:, :]
    t2 = time.time()
    print("Embedding time: {}".format(t2-t1))
    timepath = '../runningtime.txt'
    f = open(timepath, "a+")
    f.write("struc2gauss {} running time :{}\n".format(args.dataset, t2 - t1))
    f.close()
    print(embedding.shape)
    dimension = embedding.shape[1]
    columns = ['id'] + ['x_' + str(i) for i in range(embedding.shape[1])]
    ids = np.array(range(numOfNodes)).reshape((numOfNodes, 1))
    embedding = pd.DataFrame(np.concatenate([ids, embedding], axis=1), columns=columns)
    embedding.to_csv(
        '../embed/struc2gauss/clf/{}_{}.emb'.format(args.dataset, dimension),
        index=False,
        header=True)
    print('Save embeddings to ../embed/struc2gauss/clf/{}_{}.emb'.format(args.dataset,
                                                                         dimension))

    '''
    for d in range(10, 210, 10):
        embed = test_train_batch_KL_spherical(training_data, numOfNodes, d)

        embedding = embed.mu
        covar = embed.sigma

        print (embedding.shape)
        print type(embedding)
        np.savetxt('emb/usa/usa.' + str(d) +'.spherical.emb', embedding)
        np.savetxt('emb/usa/usa.' + str(d) +'.spherical.var', covar)

    '''
