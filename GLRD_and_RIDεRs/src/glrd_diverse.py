__author__ = 'pratik'

import nimfa
import numpy as np
import cvxpy as cvx
from numpy import linalg
import argparse
import mdl
import pandas as pd
import pickle as pkl
from task import Task
import random

'''
V ~ GF is estimated using NMF for a factorization rank r.

where,
V = n * f, node-feature matrix
G = n * r, node-role matrix
F = r * f, role-feature matrix

'''


def get_residual(G, F, k):
    m, n = G.shape
    p, q = F.shape
    if n != p:
        raise ValueError('Incorrect dimensions for Matrix Factorization')
    R = np.zeros((m, q))
    for idx in range(n):
        if idx == k:
            continue
        R += np.outer(G[:, idx], F[idx, :])
    return R


def glrd_diverse(V, G, F, r, err_V, err_F):
    # diversity threshold is 0.5
    for k in range(r):
        R = V - get_residual(G, F, k)  # compute residual

        # Solve for optimal G(.)(k) with diversity constraints
        F_k = F[k, :]
        x_star_G = linalg.lstsq(R.T, F_k.T, rcond=-1)[0].T
        x_G = cvx.Variable(x_star_G.shape[0])

        objective_G = cvx.Minimize(cvx.norm2(x_star_G - x_G))

        constraints_G = [x_G >= 0]
        for j in range(r):
            if j != k:
                constraints_G += [x_G.T * G[:, j] <= err_V]

        prob_G = cvx.Problem(objective_G, constraints_G)
        result = prob_G.solve(solver='SCS')
        if not result is None:
            G_k_min = np.asarray(x_G.value)
            G[:, k] = G_k_min
        else:
            print(result)

        # Solve for optimal F(k)(.) with diversity constraints
        G_k = G[:, k]
        x_star_F = linalg.lstsq(R, G_k, rcond=-1)[0]
        x_F = cvx.Variable(x_star_F.shape[0])
        objective_F = cvx.Minimize(cvx.norm2(x_star_F - x_F))

        constraints_F = [x_F >= 0]
        for j in range(r):
            if j != k:
                constraints_F += [x_F.T * F[j, :] <= err_F]

        prob_F = cvx.Problem(objective_F, constraints_F)
        result = prob_F.solve(solver='SCS')
        if not result is None:
            F_k_min = np.asarray(x_F.value)
            F[k, :] = F_k_min
        else:
            print(result)

    return G, F
    
def task(args):
    if args.dataset[0:3] == "clf":
        task = Task('CLF')
        label = np.loadtxt("../dataset/"+args.dataset+'.lbl', dtype=np.int)
        embed = pd.read_csv("../embed/GLRD-D/"+args.dataset+'.emb', encoding='utf-8', sep=',')
        embed = embed.drop(['id'],axis=1).values
        task.classfication(embed, label, split_ratio=0.7, loop=100)
    elif args.dataset[0:2] == "lp":
        task = Task('LP')
        datafile = open('../cache/{}-1.pkl'.format(args.dataset),'rb')
        graphAttr = pkl.load(datafile)
        datafile.close()
        edgeRmdList = graphAttr['edge_rmd']
        edgeRmvdList = graphAttr['edge_rmvd']
        negSampleList_train = graphAttr['edge_train_neg']
        negSampleList_test = graphAttr['edge_test_neg']
        test_sample = edgeRmvdList
        test_label = [1 for i in range(len(edgeRmvdList))]
        test_neg_label = [0 for i in range(len(negSampleList_test))]
        test_label.extend(test_neg_label)
        test_sample.extend(negSampleList_test)
        tmp_test = list(zip(test_label,test_sample))
        random.shuffle(tmp_test)
        test_label[:],test_sample[:] = zip(*tmp_test)
        test_label = np.array(test_label)
        embed = pd.read_csv("../embed/GLRD-D/"+args.dataset+'.emb', encoding='utf-8', sep=',')
        embed = embed.drop(['id'],axis=1).values
        task.link_prediction(embed, test_sample, test_label, split_ratio=0.7, method=args.lpmethod, loop=1)

if __name__ == "__main__":
    np.random.seed(1002)
    argument_parser = argparse.ArgumentParser(prog='compute glrd')
    #argument_parser.add_argument('-nf', '--node-feature', help='node-feature matrix file', required=True)
    #argument_parser.add_argument('-o', '--output-prefix', help='glrd output prefix', required=True)
    #argument_parser.add_argument('-od', '--output-dir', help='glrd output dir', required=True)
    
    argument_parser.add_argument('--dataset', nargs = '?', default = "clf/cora", help = 'dataset')
    argument_parser.add_argument('--lpmethod', nargs='?', default='Hadamard', help='binary operator')
    argument_parser.add_argument('--mdlit', type = int, default=10, help='mdl iters')
    argument_parser.add_argument('--mrole', type = int, default=10, help='min roles')
    args = argument_parser.parse_args()

    '''node_feature = args.node_feature
    out_prefix = args.output_prefix
    out_dir = args.output_dir'''
    
    node_feature_file = "../embed/ReFeX/" + args.dataset + ".emb"
    actual_fx_matrix = pd.read_csv(node_feature_file, encoding='utf-8', sep=',').drop(['id'],axis=1).values

    #refex_features = np.loadtxt(node_feature, delimiter=',')

    #np.savetxt(out_dir + '/out-' + out_prefix + '-ids.txt', X=refex_features[:, 0])

    #actual_fx_matrix = refex_features[:, 1:]

    n, f = actual_fx_matrix.shape
    print('Number of Features: ', f)
    print('Number of Nodes: ', n)

    number_bins = max(int(np.log2(n)), args.mrole)
    max_roles = min([n, f])
    best_G = None
    best_F = None

    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0
    diversity_threshold = 0.5 # fixing it to 0.5

    for rank in range(number_bins, max_roles + 1):
        lsnmf = nimfa.Lsnmf(actual_fx_matrix, rank=rank, max_iter=100)
        lsnmf_fit = lsnmf()
        G = np.asarray(lsnmf_fit.basis())
        F = np.asarray(lsnmf_fit.coef())

        G, F = glrd_diverse(V=actual_fx_matrix, G=G, F=F, r=rank, err_V=diversity_threshold, err_F=diversity_threshold)
        code_length_G = mdlo.get_huffman_code_length(G)
        code_length_F = mdlo.get_huffman_code_length(F)

        # For total bit length:
        # model_cost = code_length_W + code_length_H  # For total bit length
        # For avg. symbol bit length:
        model_cost = code_length_G * (G.shape[0] + G.shape[1]) + code_length_F * (F.shape[0] + F.shape[1])
        estimated_matrix = np.asarray(np.dot(G, F))
        loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)

        description_length = model_cost - loglikelihood

        if description_length < minimum_description_length:
            minimum_description_length = description_length
            best_G = np.copy(G)
            best_G[best_G <= 0.0] = 0.0
            best_F = np.copy(F)
            best_F[best_F <= 0.0] = 0.0
            min_des_not_changed_counter = 0
        else:
            min_des_not_changed_counter += 1
            if min_des_not_changed_counter == 10:
                break
        try:
            print('Number of Roles: {}, Model Cost: {}, -loglikelihood: {}, Description Length: {}, MDL: {}'.format(rank, model_cost, loglikelihood, description_length, minimum_description_length, best_G.shape[1]))
        except Exception:
            continue

    print('MDL has not changed for these many iters:', min_des_not_changed_counter)
    print('MDL: {}, Roles:{}'.format(minimum_description_length, best_G.shape[1]))
    embedding = best_G
    columns = ["id"] + [ "x_" +str(x) for x in range(embedding.shape[1])]
    ids = np.array([n for n in range(n)]).reshape(-1,1)
    embedding = pd.DataFrame(np.concatenate((ids, embedding), axis = 1), columns = columns)
    embedding = embedding.sort_values(by=['id'])
    print(embedding.shape)
    embedding.to_csv("../embed/GLRD-D/" + args.dataset +".emb", index = None)
    task(args)
    #np.savetxt(out_dir + '/' + 'out-' + out_prefix + "-nodeRoles.txt", X=best_G)
    #np.savetxt(out_dir + '/' + 'out-' + out_prefix + "-roleFeatures.txt", X=best_F)
