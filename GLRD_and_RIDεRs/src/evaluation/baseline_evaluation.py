# Evaluation of AAD and MSD on the node-role matrix assigned for each network by the MDL criteria
# Averaged over Multiple Random Role Assignments

import argparse
import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import normalize
from scipy.stats import powerlaw as pl
import os


def estimate_coeff(measurements, basis):
    coeff = np.zeros((measurements.shape[0], basis.shape[0]))
    for j in xrange(0, coeff.shape[0]):
        res = opt.nnls(basis.T, measurements[j, :])
        coeff[j, :] = res[0]
    return coeff


def estimate_basis(measurements, coeff):
    return estimate_coeff(measurements.T, coeff.T).T


def get_random_role_assignment(num_nodes, num_roles, seed=1000):
    random_role_assignment = np.zeros((num_nodes, num_roles))
    import random
    random.seed(seed)
    value = 1.0 / num_nodes
    for node in xrange(num_nodes):
        role = random.randint(0, num_roles - 1)
        random_role_assignment[node][role] = value
    return random_role_assignment


def get_powerlaw_random_role_assignment(num_nodes, num_roles, alpha=3.0, seed=1000):
    random_role_assignment = np.zeros((num_nodes, num_roles))
    np.random.seed(seed=seed)
    simulated_data = pl.rvs(alpha, size=num_nodes)
    hist, bins = np.histogram(simulated_data, bins=num_roles-1)
    default_value = 1.0
    test = []
    roles = np.digitize(simulated_data, bins)
    for node, role in zip(xrange(num_nodes), roles):
        test.append(role)
        random_role_assignment[node][role - 1] = default_value
    return random_role_assignment


def get_node_sense_matrix(E, E_ones):
    node_sense_matrix = []

    for r in xrange(E.shape[0]):
        m = []
        for s in xrange(E.shape[1]):
            m.append((E[r][s] / E_ones[0][s]))
        node_sense_matrix.append(m)

    return np.asarray(node_sense_matrix)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='eval aad/msd for the MDL roles')
    argument_parser.add_argument('-if', '--input-folder', help='input folder', required=True)
    argument_parser.add_argument('-nm', '--node-measurement', help='node-measure matrix file', required=True)

    args = argument_parser.parse_args()

    input_folder = args.input_folder
    node_measurement_file = args.node_measurement
    node_measurements = np.loadtxt(node_measurement_file, delimiter=',')

    methods = ['riders_r', 'riders', 'rolx', 'sparse', 'diverse']
    methods_id = {'riders_r': 'riders', 'riders': 'riders',
                  'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}

    all_measurement_labels = ['Betweenness', 'Closeness', '#BCC',
                              'Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt',
                              'Degree', 'Wt. Degree', 'Clustering Coeff']

    method_measurement_aad = np.zeros((5, 10))
    method_measurement_aad_std = np.zeros((5, 10))
    method_measurement_msd = np.zeros((5, 10))
    method_stds = {}

    for idx, method in enumerate(methods):
        print 'Method: ', method
        fname = '%s-nodeRoles.txt' % method
        fname = os.path.join(input_folder, fname)

        fname_id = 'out-%s-ids.txt' % (methods_id[method])
        fname_id = os.path.join(input_folder, fname_id)

        node_roles = np.loadtxt(fname)
        node_roles[node_roles <= 0.0] = 0.0
        try:
            n, r = node_roles.shape
        except ValueError:
            nnr = np.zeros((n, 2))
            nnr[:, 0] = node_roles
            node_roles = nnr
            n, r = node_roles.shape
        print 'Nodes: %s, Roles: %s' % (n, r)

        node_ids = np.loadtxt(fname_id)
        node_id_seq = [int(node) for node in node_ids]

        aligned_node_measurements = []
        for node_id in node_id_seq:
            try:
                aligned_node_measurements.append(node_measurements[node_id, 1:])
            except IndexError:
                print node_id, node_measurements[-1, 0]

        aligned_node_measurements = np.asarray(aligned_node_measurements)
        normalized_measurements = normalize(aligned_node_measurements, norm='l2', axis=0)

        G_ones = np.ones((aligned_node_measurements.shape[0], 1))
        E_ones = np.asarray(estimate_basis(normalized_measurements, G_ones))

        normalized_node_roles = normalize(node_roles)
        E = np.asarray(estimate_basis(normalized_measurements, normalized_node_roles))

        node_sense = get_node_sense_matrix(E, E_ones)
        print 'Node Sense Shape: ', node_sense.shape

        for i in xrange(500):
            # random_role_assignment = normalize(get_random_role_assignment(node_measurements.shape[0],
            #                                                               r, 1000 + i))
            random_role_assignment = normalize(get_powerlaw_random_role_assignment(node_measurements.shape[0],
                                                                                   r, seed=1000+i))
            E_ran = np.asarray(estimate_basis(normalized_measurements, random_role_assignment))
            random_sense = get_node_sense_matrix(E_ran, E_ones)

            for j, label in enumerate(all_measurement_labels):
                label_measurement = node_sense[:, j]
                random_label_measurement = random_sense[:, j]
                aad = np.mean(np.abs(label_measurement - random_label_measurement))
                msd = np.mean(np.square(label_measurement - random_label_measurement))
                method_measurement_aad[idx][j] += aad
                method_measurement_msd[idx][j] += msd
                if method not in method_stds:
                    method_stds[method] = {}
                    if label not in method_stds[method]:
                        method_stds[method][label] = []
                        method_stds[method][label].append(aad)
                    else:
                        method_stds[method][label].append(aad)
                else:
                    if label not in method_stds[method]:
                        method_stds[method][label] = []
                        method_stds[method][label].append(aad)
                    else:
                        method_stds[method][label].append(aad)

        print '*********************'

    method_measurement_msd /= 500.0
    method_measurement_aad /= 500.0

    np.savetxt('methods_aad_ran.txt', method_measurement_aad)

    for i, method in enumerate(methods):
        for j, label in enumerate(all_measurement_labels):
            method_measurement_aad_std[i][j] = np.std(method_stds[method][label])
    np.savetxt('methods_aad_ran_std.txt', method_measurement_aad_std)