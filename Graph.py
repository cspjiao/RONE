import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
def generate_graph(N,path):
    # 生成一个含有N个节点、每个节点有3个邻居、以概率p=0.5随机化重连边的WS小世界网络
    G = nx.random_graphs.watts_strogatz_graph(N, 3, 0.5)
    # 生成一个含有N个节点、每次加入1条边的BA无标度网络
    G = nx.random_graphs.barabasi_albert_graph(N, 3)
    #生成一个含有N个节点，每个节点有3个邻居节点的规则图。
    G = nx.random_graphs.random_regular_graph(3, N)
    #生成一个含有N个节点、以概率0.5连接的ER随机图
    G = nx.random_graphs.erdos_renyi_graph(N, 0.1)
    print(len(G.nodes()),len(G.edges()))
    df = []
    for i in G.edges():
        df.append([i[0],i[1]])
    df = pd.DataFrame(df, columns=['node1', 'node2'])
    df.to_csv(path, index=False, header=False)

for N in range(1000,11000,1000):
    #WS
    path = 'dataset/clf/random_graph_{}.edge'.format(N)
    #BA
    path = 'dataset/clf/BA_{}.edge'.format(N)
    #ER
    path = 'dataset/clf/ER_{}.edge'.format(N)
    #Regular Graph
    path = 'dataset/clf/Regular_{}.edge'.format(N)
    generate_graph(N,path)
