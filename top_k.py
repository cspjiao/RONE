# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from task import Task


def parse_args():
    parser = argparse.ArgumentParser(description="Try.")
    parser.add_argument('--dataset', nargs='?', default='clf/brazil-flights',
                        help='Input graph path')
    parser.add_argument('--lpmethod', nargs='?', default='Hadamard', help='binary operator')
    parser.add_argument('--dimension', nargs='?', default=128, type=int,
                        help='dimension of embedding')
    parser.add_argument('--attr', nargs='?', default='adj',
                        help='Graph is with attributes or not')
    parser.add_argument('--act', nargs='?', default='relu', help='activation function')
    parser.add_argument('--n_in', nargs='?', default=256, type=int, help='hidden units')
    parser.add_argument('--n_h1', nargs='?', default=256, type=int, help='hidden units')
    parser.add_argument('--n_h2', nargs='?', default=128, type=int, help='hidden units')
    parser.add_argument('--n_h3', nargs='?', default=128, type=int, help='hidden units')
    parser.add_argument('--D', nargs='?', default='F', help='distance type')
    parser.add_argument('--lr', nargs='?', default=0.001, type=float, help='learning rate')
    parser.add_argument('--l2', nargs='?', default=0.05, type=float, help='l2_coef')
    parser.add_argument('--sp', nargs='?', default=0.7, type=float, help='split ratio')
    parser.add_argument('--device', nargs='?', default=0, help='device')
    parser.add_argument('--epochs', nargs='?', default=250, type=int, help='epochs')
    parser.add_argument('--itv', nargs='?', default=5, type=int, help='epochs')
    parser.add_argument('--l', nargs='?', default=1, type=int, help='layers')
    parser.add_argument('--batch', nargs='?', default=32, type=int, help='batch size')
    parser.add_argument('--workers', nargs='?', default=4, type=int, help='workers')
    parser.add_argument('--method', nargs='?', default='ReFeX', help='Method')
    parser.add_argument('--loop', nargs='?', default=100, type=int, help='loop')
    return parser.parse_args()


def tasks(args, embed, sp=0.7, loop=10):
    task = Task('CLF')
    label = np.loadtxt("dataset/clf/" + args.dataset + '.lbl', dtype=np.int)
    result = task.classfication(embed, label, split_ratio=sp, loop=loop)
    return result


def main(args):
    task = Task('CLF')
    #datasets = ['br-wiki-talk','cy-wiki-talk','eo-wiki-talk','gl-wiki-talk','ht-wiki-talk','oc-wiki-talk']
    datasets = ['brazil-flights', 'europe-flights', 'usa-flights', 'actor', 'reality-call', 'film']
    methods = ['RolX', 'RIDeRs-S', 'GraphWave', 'SEGK', 'struc2vec', 'struc2gauss', 'role2vec', 'node2bits','DRNE', 'GraLSP', 'GAS', 'RESD']
    bots_acc_path = 'bots_acc.txt'
    admins_acc_path = 'admins_acc.txt'
    top_k = [5,10,25,50,100]
    for args.dataset in datasets:
        f1 = open(bots_acc_path, "a+")
        f2 = open(admins_acc_path, "a+")
        f1.write("{} {} ".format(args.dataset, args.method))
        f2.write("{} {} ".format(args.dataset, args.method))
        for args.method in methods:
            print("================================current method == {}================================".format(args.method))
            print("================================current dataset == {}===============================".format(args.dataset))
            try:
                embed = pd.read_csv("embed/{}/clf/{}_{}.emb".format(args.method, args.dataset,args.dimension),dtype=np.float32)
            except Exception as e:
                embed = pd.read_csv("embed/" + args.method + "/clf/" + args.dataset + '.emb',encoding='utf-8', sep=',')
            embed = embed.drop(['id'], axis=1).values
            print(embed.shape)
            for i in top_k:
                print("===============================k == {}===============================".format(i))
                df = []
                for j in range(20):
                    eval_dict = task.k_precision(embed, "dataset/clf/" + args.dataset + '.lbl', k=i)
                    df.append([args.dataset,i,j,eval_dict['precision'], eval_dict['bots_acc'], eval_dict['admins_acc']])
                df = pd.DataFrame(df, columns=['dataset', 'k', 'index', 'precision', 'bots_acc', 'admins_acc'])
                df.to_csv('embed/{}/clf/k_precision.txt'.format(args.method), index=False, header=True,mode='a')
                m1 = df.iloc[:, 4].mean()
                v1 = df.iloc[:, 4].std()
                m2 = df.iloc[:, 5].mean()
                v2 = df.iloc[:, 5].std()
                print("20 times node top-k: method = ",args.method,"dataset = ",args.dataset,"bots_acc = ",m1 ,"±",v1)
                print("20 times node top-k: method = ",args.method,"dataset = ",args.dataset,"admins_acc = ",m2 ,"±",v2)
                f1.write("{}±{} ".format(m1, v1))
                f2.write("{}±{} ".format(m2, v2))
        f1.write("\n")
        f2.write("\n")
        f1.close()
        f2.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
