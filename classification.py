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
    parser.add_argument('--loop', nargs='?', default=1, type=int, help='loop')
    return parser.parse_args()


def tasks(args, embed, sp=0.7, loop=10):
    task = Task('CLF')
    label = np.loadtxt("dataset/clf/" + args.dataset + '.lbl', dtype=np.int)
    result = task.classfication(embed, label, split_ratio=sp, loop=loop)
    return result


def main(args):
    task = Task('CLF')
    #datasets = ['br-wiki-talk','cy-wiki-talk','eo-wiki-talk','gl-wiki-talk','ht-wiki-talk','oc-wiki-talk']
    datasets = ['brazil-flights','europe-flights','usa-flights','actor','reality-call','film']
    methods = ['RolX', 'RIDeRs-S', 'GraphWave', 'SEGK', 'struc2vec', 'struc2gauss', 'role2vec', 'node2bits','DRNE', 'GraLSP', 'GAS', 'RESD']
    F1_micro_path = 'F1_micro.txt'
    F1_macro_path = 'F1_macro.txt'
    for args.method in methods:
        for args.dataset in datasets:
            print("================================current method == {}================================".format(args.method))
            print("================================current dataset == {}===============================".format(args.dataset))
            f1 = open(F1_micro_path, "a+")
            f2 = open(F1_macro_path, "a+")
            f1.write("{} {} ".format(args.method,args.dataset))
            f2.write("{} {} ".format(args.method,args.dataset))
            label = np.loadtxt("dataset/clf/" + args.dataset + '.lbl', dtype=np.int)
            try:
                embed = pd.read_csv(
                    "embed/" + args.method + "/clf/" + args.dataset + '_' + str(
                        args.dimension) + '.emb',
                    encoding='utf-8', sep=',')
            except Exception as e:
                embed = pd.read_csv(
                    "embed/" + args.method + "/clf/" + args.dataset + '.emb',
                    encoding='utf-8', sep=',')
            embed = embed.drop(['id'], axis=1).values
            print(embed.shape)

            # run10，0.7训练集，取均值和标准差classification
            for i in np.round(np.linspace(0.1, 0.9, 9), decimals=1):
                df = []
                print("================================current ratio == {}===============================".format(i))
                for j in range(20):
                    result = task.classfication(embed, label, split_ratio=i, loop=args.loop)
                    df.append([j, result['f1-micro'], result['f1-macro']])
                df = pd.DataFrame(df, columns=['index', 'Micro-F1', 'Macro-F1'])
                print("20 times node classification: train_rate = ",i,"method = ",args.method,"dataset = ",
                      args.dataset,"F1-micro = ",df.iloc[:,1].mean(),"±",df.iloc[:,1].std())
                print("20 times node classification: train_rate = ",i,"method = ",args.method,"dataset = ",
                      args.dataset,"F1-macro = ",df.iloc[:,2].mean(),"±",df.iloc[:,2].std())
                f1.write("{}±{} ".format(round(df.iloc[:,1].mean(),4),round(df.iloc[:,1].std(),4)))
                f2.write("{}±{} ".format(round(df.iloc[:,2].mean(),4),round(df.iloc[:,2].std(),4)))
        f1.write("\n")
        f2.write("\n")
        f1.close()
        f2.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
