import pickle as pkl
import time
from functools import reduce
from parser import parameter_parser
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm


def dataset_reader(path):
    if "/clf/" in path:
        edges = pd.read_csv(path, encoding='utf-8', header=None, sep=' ').values.tolist()
        graph = nx.from_edgelist(edges)
    elif "/lp/" in path:
        datafile = open(path, 'rb')
        graphAttr = pkl.load(datafile)
        graph = graphAttr['G_rmd']
        datafile.close()
    print(len(graph.nodes()), len(graph.edges()))
    return graph


def inducer(graph, node):
    nebs = list(nx.neighbors(graph, node))
    sub_nodes = nebs + [node]
    sub_g = nx.subgraph(graph, sub_nodes)
    out_counts = np.sum(list(map(lambda x: len(list(nx.neighbors(graph, x))), sub_nodes)))
    return sub_g, out_counts, nebs


def complex_aggregator(x):
    return [np.min(x), np.std(x), np.var(x), np.mean(x), np.percentile(x, 25),
            np.percentile(x, 50),
            np.percentile(x, 100), scipy.stats.skew(x), scipy.stats.kurtosis(x)]


def aggregator(x):
    return [np.sum(x), np.mean(x)]


def state_printer(x):
    print("-" * 30)
    print(x)
    print("")


def m(q, s):
    if s >= q[0] and s <= q[1]:
        return 0
    elif s > q[1] and s <= q[2]:
        return 1
    elif s > q[2] and s <= q[3]:
        return 2
    elif s > q[3] and s <= q[4]:
        return 3


def sub_selector(old_features, new_features, pruning_threshold):
    print("Cross-temporal feature pruning started.")
    indices = set()
    for i in tqdm(range(0, old_features.shape[1])):
        for j in range(0, new_features.shape[1]):
            c = np.corrcoef(old_features[:, i], new_features[:, j])
            if abs(c[0, 1]) > pruning_threshold:
                indices = indices.union(set([j]))
        keep = list(set(range(0, new_features.shape[1])).difference(indices))
        new_features = new_features[:, keep]
        indices = set()
    return new_features


class RecursiveExtractor:

    def __init__(self, args):
        self.args = args
        if self.args.aggregator == "complex":
            self.aggregator = complex_aggregator
        else:
            self.aggregator = aggregator
        self.multiplier = len(self.aggregator(0))
        input_file_path = '../dataset/' + self.args.dataset + ".edge"
        print(self.args.dataset)
        self.graph = nx.read_edgelist(input_file_path, nodetype=int)

        self.nodes = nx.nodes(self.graph)
        self.create_features()

    def basic_stat_extractor(self):
        self.base_features = []
        self.sub_graph_container = {}
        for node in tqdm(range(0, len(self.nodes))):
            sub_g, overall_counts, nebs = inducer(self.graph, node)
            in_counts = len(nx.edges(sub_g))
            self.sub_graph_container[node] = nebs
            deg = nx.degree(sub_g, node)
            trans = nx.clustering(sub_g, node)
            self.base_features.append(
                [in_counts, overall_counts, float(in_counts) / float(overall_counts),
                 float(overall_counts - in_counts) / float(overall_counts), deg, trans])
        self.features = {}
        self.features[0] = np.array(self.base_features)
        del self.base_features

    def single_recursion(self, i):
        features_from_previous_round = self.features[i].shape[1]
        new_features = np.zeros(
            (len(self.nodes), features_from_previous_round * self.multiplier))
        for k in tqdm(range(0, len(self.nodes))):
            selected_nodes = self.sub_graph_container[k]
            main_features = self.features[i][selected_nodes, :]
            new_features[k, :] = reduce(lambda x, y: x + y,
                                        [self.aggregator(main_features[:, j]) for j in
                                         range(0, features_from_previous_round)])
        return new_features

    def do_recursions(self):
        for recursion in range(0, self.args.recursive_iterations):
            state_printer("Recursion round: " + str(recursion + 1) + ".")
            new_features = self.single_recursion(recursion)
            new_features = sub_selector(self.features[recursion], new_features,
                                        self.args.pruning_cutoff)
            self.features[recursion + 1] = new_features
        #print("do_recursions self.features=========",self.features)
        self.features = np.concatenate(list(self.features.values()), axis=1)
        self.features = self.features / (np.max(self.features) - np.min(self.features))

    def binarize(self):
        self.new_features = []
        for x in tqdm(range(0, self.features.shape[1])):
            try:
                self.new_features = self.new_features + [pd.get_dummies(pd.qcut(self.features[:,
                                                                                x], self.args.bins,
                                                                                labels=range(0, self.args.bins),
                                                                                duplicates="drop"))]
            except:
                pass
        #print("self.new_features============",self.new_features)
        self.new_features = pd.concat(self.new_features, axis=1)

    def dump_to_disk(self):
        ids = {'id': [i for i in range(len(self.nodes))]}
        idpd = pd.DataFrame(ids)
        self.new_features.columns = map(lambda x: "x_" + str(x),
                                        range(0, self.new_features.shape[1]))
        self.new_features.to_csv("../cache/features/" + self.args.dataset.replace('clf/', '') + "_features.csv",
                                                index=None)
        print(self.new_features.shape)
        self.new_features.insert(0, 'id', idpd.pop('id'))
        self.new_features.to_csv("../embed/ReFeX/" + self.args.dataset + ".emb", index=None)
        print(self.new_features.shape)

    def create_features(self):
        t1 = time.time()
        print("Start time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))))
        state_printer(
            "Basic node level feature extraction and induced subgraph creation started.")
        self.basic_stat_extractor()
        state_printer("Recursion started.")
        self.do_recursions()
        state_printer("Binary feature quantization started.")
        self.binarize()
        t2 = time.time()
        print("running ReFeX time: {}".format(t2 - t1))
        timepath='../runningtime.txt'
        f=open(timepath,"a+")
        f.write("ReFeX {} running time :{}\n".format(self.args.dataset,t2-t1))
        f.close()
        state_printer("Saving the raw features.")
        self.dump_to_disk()
        state_printer(
            "The number of extracted features is: " + str(self.new_features.shape[1]) + ".")

if __name__ == '__main__':
    start = time.time()
    model = RecursiveExtractor(parameter_parser())
    print("Total time: %s" % (time.time() - start))

