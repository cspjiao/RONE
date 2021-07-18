__author__ = 'pratik'

from collections import defaultdict
import networkx as nx
import numpy as np
import sys
import pickle


def get_egonet_members(graph, vertex, level=0):
    """
    node's egonet members, supports level 0 and level 1 egonet
    :param vertex: vertex_id
    :param level: level <level> egonet. Default 0
    :return: returns level <level> egonet of the vertex
    """
    lvl_zero_egonet = graph.neighbors(vertex)
    lvl_zero_egonet.append(vertex)
    if level == 1:
        lvl_one_egonet = []
        for node in lvl_zero_egonet:
            if node != vertex:
                lvl_one_egonet.extend(graph.neighbors(node))
        lvl_zero_egonet.extend(lvl_one_egonet)
    return list(set(lvl_zero_egonet))


try:
    input_file = sys.argv[1]
    out_dir = sys.argv[2]
except IndexError:
    print 'usage: python %s <graph_file> <output_dir>' % sys.argv[0]
    sys.exit(1)

graph = nx.Graph()

for line in open(input_file):
    line = line.strip().split(',')
    s = int(line[0])
    d = int(line[1])
    w = float(line[2])
    graph.add_edge(s, d, weight=w)

betweenness = nx.betweenness_centrality(graph, normalized=True)
closeness = nx.closeness_centrality(graph, normalized=True)
clustering_coeff = nx.clustering(graph)
degree = nx.degree_centrality(graph)

components = nx.biconnected_components(graph)
biconn = defaultdict(int)
for component in components:
    for node in component:
        biconn[node] += 1

weighted_degree = defaultdict(int)
nodes = graph.nodes()
for node in nodes:
    adj_list = graph.neighbors(node)
    for neighbor in adj_list:
        weighted_degree[node] += graph[node][neighbor]['weight']

egonet_zero_degree = defaultdict(float)
egonet_zero_weight = defaultdict(float)

egonet_one_degree = defaultdict(float)
egonet_one_weight = defaultdict(float)

for node in graph.nodes():
    node_egonet_zero = get_egonet_members(graph, node, level=0)
    node_egonet_one = get_egonet_members(graph, node, level=1)

    for ego_member in node_egonet_zero:
        egonet_zero_degree[node] += degree[ego_member]
        egonet_zero_weight[node] += weighted_degree[ego_member]

    for ego_member in node_egonet_one:
        egonet_one_degree[node] += degree[ego_member]
        egonet_one_weight[node] += weighted_degree[ego_member]


measurement_matrix = []
for node in sorted(nodes):
    node_measurements = [node]  # 0
    node_measurements.append(betweenness[node])  # 1
    node_measurements.append(closeness[node])  # 2
    node_measurements.append(biconn[node])  # 3
    node_measurements.append(egonet_zero_degree[node])  # 4
    node_measurements.append(egonet_one_degree[node])  # 5
    node_measurements.append(egonet_zero_weight[node])  # 6
    node_measurements.append(egonet_one_weight[node])  # 7
    node_measurements.append(degree[node])  # 8
    node_measurements.append(weighted_degree[node])  # 9
    node_measurements.append(clustering_coeff[node])  # 10
    measurement_matrix.append(node_measurements)

pickle.dump(betweenness, open(out_dir + '/betweenness.p', 'wb'))
pickle.dump(closeness, open(out_dir + '/closeness.p', 'wb'))
pickle.dump(biconn, open(out_dir + '/biconn.p', 'wb'))
pickle.dump(egonet_zero_degree, open(out_dir + '/ego_zero_deg.p', 'wb'))
pickle.dump(egonet_one_degree, open(out_dir + '/ego_one_deg.p', 'wb'))
pickle.dump(egonet_zero_weight, open(out_dir + '/ego_zero_weight.p', 'wb'))
pickle.dump(egonet_one_weight, open(out_dir + '/ego_one_weight.p', 'wb'))
pickle.dump(degree, open(out_dir + '/degree.p', 'wb'))
pickle.dump(clustering_coeff, open(out_dir + '/clustering_coeff.p', 'wb'))
pickle.dump(weighted_degree, open(out_dir + '/weighted_degree.p', 'wb'))
np.savetxt(out_dir + '/measurements.txt', np.asarray(measurement_matrix), delimiter=',')

print '*'*50
print 'Finished: ', out_dir
print '*'*50

