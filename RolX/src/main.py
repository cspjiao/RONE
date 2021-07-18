import time
from parser import parameter_parser
from rolx import ROLX
from task import Task
import numpy as np
import pickle as pkl
import pandas as pd
import random
import time

def create_and_run_model(args):
    """
    Function to read the graph, create features and train the embedding.
    """
    t1 = time.time()
    print("Start time: {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))))
    model = ROLX(args)
    if args.skip != 0:
        embeddings = model.train()
        t2 = time.time()
        print("all running time: {}".format(t2 - t1))
        timepath = '../runningtime.txt'
        f = open(timepath, "a+")
        f.write("RolX {} running time :{}\n".format(args.dataset, t2 - t1))
        f.close()
        columns = ["id"] + ["x_" + str(x) for x in range(embeddings.shape[1])]
        ids = np.array(list(range(embeddings.shape[0]))).reshape((-1, 1))
        embedding = pd.DataFrame(np.concatenate([ids, embeddings], axis=1), columns=columns)
        embedding = embedding.sort_values(by=['id'])
        print('Save best embedding to ../embed/RolX/{}_{}.emb'.format(args.dataset,
                                                                      args.dimensions))
        embedding.to_csv("../embed/RolX/{}_{}.emb".format(args.dataset, args.dimensions),
                     index=False)

def tasks(method, args):
    if method == 'ReFeX':
        pre = "../embed/ReFeX/"
    elif method == 'RolX':
        pre = "../embed/RolX/"
    if args.dataset[0:3] == "clf":
        task = Task('CLF')
        label = np.loadtxt("../dataset/" + args.dataset + '.lbl', dtype=np.int)
        if method == 'RolX':
            embed = pd.read_csv(pre + args.dataset + '_' + str(args.dimensions) + '.emb',
                                encoding='utf-8', sep=',')
        else:
            embed = pd.read_csv(pre + args.dataset + '.emb',
                                encoding='utf-8', sep=',')
        embed = embed.drop(['id'], axis=1).values
        return task.classfication(embed, label, split_ratio=0.7, loop=20)
    elif args.dataset[0:2] == "lp":
        task = Task('LP')
        datafile = open('../cache/{}-1.pkl'.format(args.dataset), 'rb')
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
        tmp_test = list(zip(test_label, test_sample))
        random.shuffle(tmp_test)
        test_label[:], test_sample[:] = zip(*tmp_test)
        test_label = np.array(test_label)
        embed = pd.read_csv(pre + args.dataset + '.emb', encoding='utf-8', sep=',')
        embed = embed.drop(['id'], axis=1).values
        return task.link_prediction(embed, test_sample, test_label, split_ratio=0.7,
                                    method=args.lpmethod, loop=1)


if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)
    # if args.skip == 0:
    #     eval1 = tasks('ReFeX', args)
    # else:
    #     eval2 = tasks('RolX', args)
    # if args.skip == 0:
    #     print(eval1)
    # else:
    #     print(eval2)
