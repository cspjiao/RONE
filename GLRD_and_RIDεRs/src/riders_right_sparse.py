__author__ = 'pratik'
import time
import numpy as np
import argparse
import mdl
import pandas as pd
import pickle as pkl
from task import Task
import random
import nimfa
def task(args):
    if args.dataset[0:3] == "clf":
        task = Task('CLF')
        label = np.loadtxt("../dataset/"+args.dataset+'.lbl', dtype=np.int)
        embed = pd.read_csv("../embed/RIDeRs-S/"+args.dataset+'.emb', encoding='utf-8', sep=',')
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
        embed = pd.read_csv("../embed/RIDeRs-S/"+args.dataset+'.emb', encoding='utf-8', sep=',')
        embed = embed.drop(['id'],axis=1).values
        task.link_prediction(embed, test_sample, test_label, split_ratio=0.7, method=args.lpmethod, loop=1)


if __name__ == "__main__":
    np.random.seed(1000)
    argument_parser = argparse.ArgumentParser(prog='compute right sparsity')
    #argument_parser.add_argument('-nf', '--node-feature', help='node-feature matrix file', required=True)
    #argument_parser.add_argument('-o', '--output-prefix', help='glrd output prefix', required=True)
    #argument_parser.add_argument('-od', '--output-dir', help='glrd output dir', required=True)
    
    argument_parser.add_argument('--dataset', nargs = '?', default = "clf/brazil-flights", help = 'dataset')
    argument_parser.add_argument('--lpmethod', nargs='?', default='Hadamard', help='binary operator')
    argument_parser.add_argument('--mdlit', type = int, default=10, help='mdl iters')
    argument_parser.add_argument('--mrole', type = int, default=10, help='min roles')
    args = argument_parser.parse_args()

    '''node_feature = args.node_feature
    out_prefix = args.output_prefix
    out_dir = args.output_dir'''
    t1 = time.time()
    print("Start time: {}".format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))))
    try:
        actual_fx_matrix = pd.read_csv("../embed/ReFeX/" + args.dataset + "_128.emb",encoding='utf-8', sep=',')
    except Exception as e:
        actual_fx_matrix = pd.read_csv("../embed/ReFeX/" + args.dataset + '.emb',encoding='utf-8', sep=',')
    actual_fx_matrix = actual_fx_matrix.drop(['id'],axis=1).values
    '''dataname = args.dataset.replace('clf/','').replace('lp/','')
    node_feature_file = "./rider/{}/feature-matrix.txt".format(dataname)
    actual_fx_matrix = pd.read_csv(node_feature_file, encoding='utf-8', sep=',', header=None).values'''
    '''refex_features = np.loadtxt(node_feature, delimiter=',')
    np.savetxt(out_dir + '/out-' + out_prefix + '-ids.txt', X=refex_features[:, 0])
    actual_fx_matrix = refex_features[:, 1:]'''

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
    sparsity_threshold = 1.0
    print("number_bins = ",number_bins)
    print("\nmax_roles = ",max_roles)
    print("\n")
    for rank in range(number_bins, max_roles + 1):
        snmf = nimfa.Snmf(actual_fx_matrix, seed="random_vcol", version='r', rank=rank, beta=2.0)
        snmf_fit = snmf()
        G = np.asarray(snmf_fit.basis())
        F = np.asarray(snmf_fit.coef())

        code_length_G = mdlo.get_huffman_code_length(G)
        code_length_F = mdlo.get_huffman_code_length(F)

        model_cost = code_length_G * (G.shape[0] + G.shape[1]) + code_length_F * (F.shape[0] + F.shape[1])
        estimated_matrix = np.asarray(np.dot(G, F))
        loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)
        err = snmf_fit.distance(metric='kl')

        description_length = model_cost + err  #- loglikelihood

        if description_length < minimum_description_length:
            minimum_description_length = description_length
            best_G = np.copy(G)
            best_F = np.copy(F)
            min_des_not_changed_counter = 0
        else:
            min_des_not_changed_counter += 1
            if min_des_not_changed_counter == args.mdlit:
                break
        #print('Number of Roles: {}, Model Cost: {}, -loglikelihood: {}, Description Length: {}, MDL: {}'.format(rank, model_cost, loglikelihood, description_length, minimum_description_length, best_G.shape[1]))
        try:
            print('Number of Roles: {}, Model Cost: {}, -loglikelihood: {}, Description Length: {}, MDL: {}'.format(rank, model_cost, loglikelihood, description_length, minimum_description_length, best_G.shape[1]))
        except Exception:
            continue
    print('MDL has not changed for these many iters:', min_des_not_changed_counter)
    print('\nMDL: {}, Roles: {}'.format(minimum_description_length, best_G.shape[1]))
    embedding = best_G
    t2 = time.time()
    print("Embedding time: {}".format(t2 - t1))
    timepath = '../runningtime.txt'
    f = open(timepath, "a+")
    f.write("RIDeRs {} running time :{}\n".format(args.dataset, t2 - t1))
    f.close()
    columns = ["id"] + [ "x_" +str(x) for x in range(embedding.shape[1])]
    ids = np.array([n for n in range(n)]).reshape(-1,1)
    embedding = pd.DataFrame(np.concatenate((ids, embedding), axis = 1), columns = columns)
    embedding = embedding.sort_values(by=['id'])
    print(embedding.shape)
    embedding.to_csv("../embed/RIDeRs-S/" + args.dataset +".emb", index = None)
    #task(args)
    #np.savetxt(out_dir + '/' + 'out-' + out_prefix + "-nodeRoles.txt", X=best_G)
    #np.savetxt(out_dir + '/' + 'out-' + out_prefix + "-roleFeatures.txt", X=best_F)
