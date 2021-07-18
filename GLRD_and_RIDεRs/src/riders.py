import features
import mdl
import argparse
import numpy as np
import nimfa
from collections import defaultdict
import pandas as pd
import pickle as pkl
from task import Task
import random

if __name__ == "__main__":
    np.random.seed(1004)
    argument_parser = argparse.ArgumentParser(prog='compute riders matrix')
    #argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    argument_parser.add_argument('--dataset', nargs = '?', default = "clf/brazil-flights", help = 'dataset')
    argument_parser.add_argument('--lpmethod', nargs='?', default='Hadamard', help='binary operator')
    #argument_parser.add_argument('-b', '--bins', help='bins for rider features', required=True)
    #argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)
    argument_parser.add_argument('--rider-dir', nargs='?', default='./rider/', help='rider directory')
    argument_parser.add_argument('--bins', type = int, default=15, help='bins for rider features')
    #argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)
    #argument_parser.add_argument('-od', '--output-dir', help='final output dir', required=True)
    #argument_parser.add_argument('-p', '--output-prefix', help='prefix', required=True)
    #argument_parser.add_argument('-rec', '--recursive-rider', help='recursive rider features', action='store_true')
    argument_parser.add_argument('-rec', '--recursive-rider', help='recursive rider features', default=True)
    args = argument_parser.parse_args()

    if 'clf/' in args.dataset:
        graph_file = "../dataset/" + args.dataset + ".edge"
    elif 'lp/' in args.dataset:
        graph_file = "../cache/" + args.dataset + "-1.pkl"
    #graph_file = args.graph
    #rider_dir = args.rider_dir
    rider_dir = "../embed/ReFeX/" + args.dataset + ".emb"
    bins = int(args.bins)
    #out_dir = args.output_dir
    #prefix = args.output_prefix
    recursive_rider_fx = args.recursive_rider

    fx = features.Features()
    fx.load_graph(graph_file)
    #fx.compute_primitive_features()
    if recursive_rider_fx:
        rider_features = fx.only_riders(graph_file=graph_file, rider_dir=rider_dir, bins=bins, bin_features=True)
        fx.prune_riders_fx_and_reassign_to_graph(rider_features)
        fx.init_vertex_egonet()
        primitive_riders_fx_matrix = fx.create_initial_feature_matrix()

        prev_pruned_fx_matrix = primitive_riders_fx_matrix

        prev_pruned_fx_size = len(list(prev_pruned_fx_matrix.dtype.names))
        no_iterations = 0
        max_diff = 1.0

        while no_iterations <= fx.MAX_ITERATIONS:
            current_iteration_pruned_fx_matrix = fx.compute_recursive_features(prev_fx_matrix=prev_pruned_fx_matrix,
                                                                               iter_no=no_iterations, max_dist=max_diff)

            if current_iteration_pruned_fx_matrix is None:
                print('No new features added, all pruned. Exiting!')
                break

            current_pruned_fx_size = len(list(current_iteration_pruned_fx_matrix.dtype.names))

            print('Iteration: %s, Number of Features: %s' % (no_iterations, current_pruned_fx_size))

            if current_pruned_fx_size == prev_pruned_fx_size:
                print('No new features added, Exiting!')
                break

            # update the latest feature matrix to the graph
            fx.update_feature_matrix_to_graph(current_iteration_pruned_fx_matrix)

            # update the previous iteration feature matrix with the latest one
            prev_pruned_fx_matrix = current_iteration_pruned_fx_matrix
            prev_pruned_fx_size = current_pruned_fx_size

            # increment feature similarity threshold by 1
            max_diff += 1.0
            no_iterations += 1

        fx_names = fx.get_current_fx_names()
        rider_features = defaultdict(list)
        graph_nodes = sorted(fx.graph.nodes())

        for node in graph_nodes:
            feature_row = []
            for fx_name in fx_names:
                feature_row.append(fx.graph.node[node][fx_name])
            rider_features[node] = feature_row
    else:
        rider_features = fx.only_riders_as_dict(graph_file=graph_file, rider_dir=rider_dir, bins=bins, bin_features=True)

    actual_fx_matrix = []
    for node in sorted(rider_features.keys()):
        actual_fx_matrix.append(rider_features[node])

    actual_fx_matrix = np.array(actual_fx_matrix)

    n, f = actual_fx_matrix.shape
    print('Number of Features: ', f)
    print('Number of Nodes: ', n)

    fx_matrix_with_node_ids = np.zeros((n, f+1))
    fx_matrix_with_node_ids[:, 0] = np.array([float(node) for node in range(n)])
    fx_matrix_with_node_ids[:, 1:] = actual_fx_matrix
    np.savetxt(out_dir + '/out-' + prefix + '-featureValues.csv', X=fx_matrix_with_node_ids, delimiter=',')
    np.savetxt(out_dir + '/out-' + prefix + '-ids.txt', X=fx_matrix_with_node_ids[:, 0])

    number_bins = int(np.log2(n))
    max_roles = min([n, f])
    best_W = None
    best_H = None

    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0

    for rank in range(1, max_roles + 1):
        lsnmf = nimfa.Lsnmf(actual_fx_matrix, rank=rank, max_iter=1000)
        lsnmf_fit = lsnmf()
        W = np.asarray(lsnmf_fit.basis())
        H = np.asarray(lsnmf_fit.coef())
        estimated_matrix = np.asarray(np.dot(W, H))

        code_length_W = mdlo.get_huffman_code_length(W)
        code_length_H = mdlo.get_huffman_code_length(H)

        model_cost = code_length_W * (W.shape[0] + W.shape[1]) + code_length_H * (H.shape[0] + H.shape[1])
        loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)

        description_length = model_cost - loglikelihood

        if description_length < minimum_description_length:
            minimum_description_length = description_length
            best_W = np.copy(W)
            best_H = np.copy(H)
            min_des_not_changed_counter = 0
        else:
            min_des_not_changed_counter += 1
            if min_des_not_changed_counter == 10:
                break

        print('Number of Roles: {}, Model Cost: {}, -loglikelihood: {}, Description Length: {}, MDL: {}'.format(rank, model_cost, loglikelihood, description_length, minimum_description_length, best_W.shape[1]))

    print('MDL has not changed for these many iters:', min_des_not_changed_counter)
    print('\nMDL: {}, Roles: {}'.format(minimum_description_length, best_W.shape[1]))
    #np.savetxt(out_dir + '/out-' + prefix + "-nodeRoles.txt", X=best_W)
    #np.savetxt(out_dir + '/out-' + prefix + "-roleFeatures.txt", X=best_H)
