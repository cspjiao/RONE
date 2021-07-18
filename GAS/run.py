# -*- coding: utf-8 -*-
from models import GAE
from preprocess import load_data, normalize_adj, normalize_adj1
import argparse
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from task import Task
from roleData import roleData
from torch.utils.data import DataLoader
import time
import pandas as pd 

def  parse_args():
    parser = argparse.ArgumentParser(description="Try.")
    parser.add_argument('--dataset', nargs='?', default='clf/brazil-flights',help='Input graph path')
    parser.add_argument('--lpmethod', nargs='?', default='Hadamard', help='binary operator')
    parser.add_argument('--attr', nargs='?', default='adj', help='Graph is with attributes or not')
    parser.add_argument('--act', nargs='?', default='sigmoid',help='activation function')
    parser.add_argument('--n_in', nargs='?', default=256,type=int,help='hidden units')
    parser.add_argument('--n_h1', nargs='?', default=256,type=int,help='hidden units')
    parser.add_argument('--n_h2', nargs='?', default=128,type=int,help='hidden units')
    parser.add_argument('--n_h3', nargs='?', default=128,type=int,help='hidden units')
    parser.add_argument('--D', nargs='?', default='F',help='distance type')
    parser.add_argument('--lr', nargs='?', default=0.001,type=float,help='learning rate')
    parser.add_argument('--l2', nargs='?', default=0.5,type=float,help='l2_coef')
    parser.add_argument('--sp', nargs='?', default=0.7,type=float,help='split ratio')
    parser.add_argument('--device', nargs='?', default=0,help='device')
    parser.add_argument('--epochs', nargs='?', default=150,type=int,help='epochs')
    parser.add_argument('--itv', nargs='?', default=5,type=int,help='epochs')
    parser.add_argument('--adjnorm', nargs='?', default=0,type=int,help='layers')
    parser.add_argument('--batch', nargs='?', default=32,type=int,help='batch size')
    parser.add_argument('--workers', nargs='?', default=4,type=int,help='workers')
    return parser.parse_args()

def tasks(args, embed):
    if args.dataset[0:3] == "clf":
        task = Task('CLF')
        label = np.loadtxt("../dataset/"+args.dataset+'.lbl', dtype=np.int)
        return task.classfication(embed, label, split_ratio=args.sp, loop=50)

def main(args):
    t1 = time.time()
    print("Start time: {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))))
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
    graph, Dist, adj_m, Degree = load_data(args)
    e = Dist['F']
    F_index = list(range(21))  #[0,1,4,15,16,17,18,19]  list(range(7))+list(range(14,21))
    args.n_h4 = len(F_index)
    print(args.n_h4)
    F = torch.FloatTensor(e[:,F_index][np.newaxis]).to(device=device, dtype=torch.float32)
    f1 = F.squeeze().data.cpu().numpy()
    columns = ["id"] + [ "x_" +str(x) for x in range(f1.shape[1])]
    ids = np.array([node for node in graph.nodes()]).reshape(-1,1)
    embedding = pd.DataFrame(np.concatenate([ids, f1], axis = 1), columns = columns)
    embedding = embedding.sort_values(by=['id'])
    embedding.to_csv("../embed/GAS/Features/" + args.dataset +".emb", index = None)

    X = torch.FloatTensor((adj_m+np.eye(adj_m.shape[0]))[np.newaxis]).to(device=device, dtype=torch.float32)
    #D = torch.FloatTensor((Degree)[np.newaxis]).to(device=device, dtype=torch.float32)
    if args.adjnorm == 0:
        adj = torch.FloatTensor((adj_m+np.eye(adj_m.shape[0]))[np.newaxis]).to(device=device, dtype=torch.float32)
        #deg = torch.FloatTensor((Degree)[np.newaxis]).to(device=device, dtype=torch.float32)
        #deg = D
    elif args.adjnorm == 2:
        adj = torch.FloatTensor(normalize_adj(adj_m+np.eye(adj_m.shape[0]))[np.newaxis]).to(device=device, dtype=torch.float32)
    elif args.adjnorm == 1:
        adj = torch.FloatTensor(normalize_adj1(adj_m+np.eye(adj_m.shape[0]))[np.newaxis]).to(device=device, dtype=torch.float32)
    print('Data Loaded!')
    args.n_in = X.shape[1]
    #args.n_in = D.shape[1]
    model = GAE(args, device).to(device=device, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    #tasks(args,e[:,F_index])
    best = 1e9
    best_t = 0
    patience = 15
    print('Start Training...')
    for epoch in range(args.epochs):    
        start_time = time.time()
        opt.zero_grad()
        loss = model(X, adj, F)
        #loss = model(D, deg, F)
        loss.backward()
        opt.step()
        print('Epoch: ',epoch,'  Loss:', loss, ' time:', time.time()-start_time)
        '''
        if (epoch + 1) % args.itv == 0:
            embed = model.embed(X,adj)
            start_time = time.time()
            tasks(args,embed)
            print(time.time() - start_time)
        '''
        if abs(loss - best) / best > 0.01:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break
        best = loss

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi.pkl'))
    embeddings = model.embed(X, adj)
    #embeddings = model.embed(D, deg)
    t2 = time.time()
    print("Embedding time: {}".format(t2 - t1))
    timepath = '../runningtime.txt'
    f = open(timepath, "a+")
    f.write("GAS {} running time :{}\n".format(args.dataset, t2 - t1))
    f.close()
    columns = ["id"] + ["x_" + str(x) for x in range(embeddings.shape[1])]
    ids = np.array(list(range(embeddings.shape[0]))).reshape((-1, 1))
    embedding = pd.DataFrame(np.concatenate([ids, embeddings], axis=1), columns=columns)
    embedding = embedding.sort_values(by=['id'])
    print('Save best embedding to ../embed/GAS/{}_{}.emb'.format(args.dataset,args.n_h2))
    embedding.to_csv("../embed/GAS/{}_{}.emb".format(args.dataset,args.n_h2),
                     index=False)
    #tasks(args,embeddings)


if __name__ == "__main__":
    args = parse_args()
    main(args)
