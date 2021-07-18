import features
import argparse
import numpy as np
import pandas as pd
import pickle as pkl

if __name__ == "__main__":
    np.random.seed(1004)
    argument_parser = argparse.ArgumentParser(prog='init features')
    argument_parser.add_argument('--dataset', nargs = '?', default = "clf/brazil-flights", help = 'dataset')
    args = argument_parser.parse_args()
    
    if 'clf/' in args.dataset:
        graph_file = "../dataset/" + args.dataset + ".edge"
    elif 'lp/' in args.dataset:
        graph_file = "../cache/" + args.dataset + "-1.pkl"
    dataname = args.dataset.replace('clf/','').replace('lp/','')
    #print(dataname)
    fx = features.Features()
    fx.load_graph(graph_file)
    fx.compute_primitive_features()
    fx.save_feature_matrix(dataname)