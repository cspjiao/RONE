import networkx as nx
import os
import numpy as np
from numpy.lib.recfunctions import merge_arrays
from collections import defaultdict
import pandas as pd
import pickle as pkl
from task import Task
import random


class Features:
    """
    The features class. Placeholder for the graph, egonet and node features.
    """

    def __init__(self):
        """
        initializes an empty networkx directed graph
        :return:
        """
        self.graph = nx.DiGraph()
        # self.graph = nx.Graph()
        self.no_of_vertices = 0
        self.p = 0.5  # fraction of nodes placed in log bin
        self.s = 0  # feature similarity threshold
        self.TOLERANCE = 0.0001
        self.MAX_ITERATIONS = 100
        self.refex_log_binned_buckets = []
        self.vertex_egonets = {}
        self.memo_recursive_fx_names = {}

    '''def load_graph(self, path):
        """
        loads a networkx DiGraph from a file. Support directed weighted graph. Each comma 
        separated line is source, destination and weight.
        :param file_name: <file_name>
        :return:
        """
        for line in open(file_name):
            line = line.strip()
            line = line.split(',')
            source = int(line[0])
            dest = int(line[1])
            wt = float(line[2])
            self.graph.add_edge(source, dest, weight=wt)'''

    def load_graph(self, path):
        if "/clf/" in path:
            G = nx.from_edgelist(pd.read_csv(path, encoding='utf-8', header=None,
                                             sep=' ').values.tolist())
            G.remove_edges_from(self.graph.selfloop_edges())
        elif "/lp/" in path:
            datafile = open(path, 'rb')
            graphAttr = pkl.load(datafile)
            G = graphAttr['G_rmd']
            datafile.close()
        self.graph.add_edges_from(G.edges())
        for s, t in self.graph.edges():
            self.graph[s][t]['weight'] = 1.0
        print(len(self.graph.nodes()), len(self.graph.edges()))

    def load_graph_with_fixed_vertices(self, file_name, num_nodes):
        """
        loads a networkx DiGraph from a file. Support directed weighted graph. Each comma
        separated line is source, destination and weight.
        :param file_name: <file_name>
        :return:
        """
        for line in open(file_name):
            line = line.strip()
            line = line.split(',')
            source = int(line[0])
            dest = int(line[1])
            wt = float(line[2])
            self.graph.add_edge(source, dest, weight=wt)

        nodes = self.graph.nodes()
        for n in range(0, num_nodes):
            if n not in nodes:
                self.graph.add_node(n)

    def get_egonet_members(self, vertex, level=0):
        """
        node's egonet members, supports level 0 and level 1 egonet
        :param vertex: vertex_id
        :param level: level <level> egonet. Default 0
        :return: returns level <level> egonet of the vertex
        """
        lvl_zero_egonet = list(self.graph.successors(vertex))
        lvl_zero_egonet.append(vertex)
        if level == 1:
            lvl_one_egonet = []
            for node in lvl_zero_egonet:
                if node != vertex:
                    lvl_one_egonet.extend(self.graph.successors(node))
            lvl_zero_egonet.extend(lvl_one_egonet)
        return list(set(lvl_zero_egonet))

    def compute_base_egonet_primitive_features(self, vertex, attrs, level_id='0'):
        """
            * Counts several egonet properties. In particular:
            *
            * wn - Within Node - # of nodes in egonet
            * weu - Within Edge Unique - # of unique edges with both ends in egonet
            * wet - Within Edge Total - total # of internal edges
            * xesu - eXternal Edge Source Unique - # of unique edges exiting egonet
            * xest - eXternal Edge Source Total - total # of edges exiting egonet
            * xedu - eXternal Edge Destination Unique - # of unique edges entering egonet
            * xedt - eXternal Edge Destination Total - total # of edges entering egonet
            *
            * and three counts per attribute,
            *
            * wea-ATTRNAME  - Within Edge Attribute - sum of attribute for internal edges
            * xea-ATTRNAME  - sum of xeda and xesa
            * xesa-ATTRNAME - eXternal Edge Source Attribute - sum of attr for exiting edges
            * xeda-ATTRNAME - eXternal Edge Destination Attribute - sum of attr for entering
            edges
            *
        :return: side effecting code, adds the features in the networkx DiGraph dict
        """
        self.graph.node[vertex]['wn' + level_id] = 0.0
        self.graph.node[vertex]['weu' + level_id] = 0.0
        self.graph.node[vertex]['wet' + level_id] = 0.0
        self.graph.node[vertex]['wea-' + level_id] = 0.0
        self.graph.node[vertex]['xedu' + level_id] = 0.0
        self.graph.node[vertex]['xedt' + level_id] = 0.0
        self.graph.node[vertex]['xesu' + level_id] = 0.0
        self.graph.node[vertex]['xest' + level_id] = 0.0
        self.graph.node[vertex]['xeu' + level_id] = 0.0
        self.graph.node[vertex]['xet' + level_id] = 0.0
        for attr in attrs:
            self.graph.node[vertex]['wea-' + attr + level_id] = 0.0
            self.graph.node[vertex]['xesa-' + attr + level_id] = 0.0
            self.graph.node[vertex]['xeda-' + attr + level_id] = 0.0
            self.graph.node[vertex]['xea-' + attr + level_id] = 0.0

        if level_id == '0':
            egonet = self.vertex_egonets[vertex][0]
        else:
            egonet = self.vertex_egonets[vertex][1]

        for n1 in egonet:
            in_neighbours = self.graph.predecessors(n1)
            out_neighbours = self.graph.successors(n1)

            self.graph.node[vertex]['wn' + level_id] += 1.0

            for n2 in in_neighbours:
                if n2 in egonet:
                    self.graph.node[vertex]['weu' + level_id] += 1.0
                    self.graph.node[vertex]['wet' + level_id] += len(
                        list(self.graph.predecessors(n2)))
                    for attr in attrs:
                        if attr == 'wgt':
                            self.graph.node[vertex]['wea-' + attr + level_id] += \
                            self.graph[n2][n1]['weight']
                else:
                    self.graph.node[vertex]['xedu' + level_id] += 1.0
                    self.graph.node[vertex]['xedt' + level_id] += len(
                        list(self.graph.predecessors(n2)))
                    for attr in attrs:
                        if attr == 'wgt':
                            self.graph.node[vertex]['xeda-' + attr + level_id] += \
                            self.graph[n2][n1]['weight']

            for n2 in out_neighbours:
                if n2 not in egonet:
                    self.graph.node[vertex]['xesu' + level_id] += 1.0
                    self.graph.node[vertex]['xest' + level_id] += len(
                        list(self.graph.successors(n2)))
                    for attr in attrs:
                        if attr == 'wgt':
                            self.graph.node[vertex]['xesa-' + attr + level_id] += \
                            self.graph[n1][n2]['weight']
                else:
                    # weu, wet and wea have already been counted as in_neighbours in some
                    # egonet
                    # do nothing
                    continue

        self.graph.node[vertex]['xeu' + level_id] = self.graph.node[vertex][
                                                        'xesu' + level_id] + \
                                                    self.graph.node[vertex]['xedu' + level_id]
        self.graph.node[vertex]['xet' + level_id] = self.graph.node[vertex][
                                                        'xest' + level_id] + \
                                                    self.graph.node[vertex]['xedt' + level_id]

        for attr in attrs:
            self.graph.node[vertex]['xea-' + attr + level_id] = self.graph.node[vertex][
                                                                    'xesa-' + attr +
                                                                    level_id] + \
                                                                self.graph.node[vertex][
                                                                    'xeda-' + attr + level_id]

    def init_rider_features(self, vertex, fx_name_base, attrs=['wgt']):
        self.graph.node[vertex]['wd-' + fx_name_base] = 0.0  # destination
        self.graph.node[vertex]['ws-' + fx_name_base] = 0.0  # source
        for attr in attrs:
            self.graph.node[vertex]['wda-' + attr + '-' + fx_name_base] = 0.0
            self.graph.node[vertex]['wsa-' + attr + '-' + fx_name_base] = 0.0

        ## Egonet Rider Features
        self.graph.node[vertex]['wes0-' + fx_name_base] = 0.0
        self.graph.node[vertex]['wes1-' + fx_name_base] = 0.0
        self.graph.node[vertex]['wed0-' + fx_name_base] = 0.0
        self.graph.node[vertex]['wed1-' + fx_name_base] = 0.0
        self.graph.node[vertex]['xes0-' + fx_name_base] = 0.0
        self.graph.node[vertex]['xes1-' + fx_name_base] = 0.0
        self.graph.node[vertex]['xed0-' + fx_name_base] = 0.0
        self.graph.node[vertex]['xed1-' + fx_name_base] = 0.0
        for attr in attrs:
            self.graph.node[vertex]['wesa-' + attr + '0-' + fx_name_base] = 0.0
            self.graph.node[vertex]['xesa-' + attr + '0-' + fx_name_base] = 0.0
            self.graph.node[vertex]['weda-' + attr + '0-' + fx_name_base] = 0.0
            self.graph.node[vertex]['xeda-' + attr + '0-' + fx_name_base] = 0.0
            self.graph.node[vertex]['wesa-' + attr + '1-' + fx_name_base] = 0.0
            self.graph.node[vertex]['xesa-' + attr + '1-' + fx_name_base] = 0.0
            self.graph.node[vertex]['weda-' + attr + '1-' + fx_name_base] = 0.0
            self.graph.node[vertex]['xeda-' + attr + '1-' + fx_name_base] = 0.0

    def digitize(self, block_size, log_bins, file_name):
        # block_size_value IS NOT the log10(block_size)
        # returns the feature_name corresponding to this block size assigned bin
        start = log_bins[0]
        i = 0
        for curr_bin in log_bins[1:]:
            if start <= block_size < curr_bin:
                return file_name + '_' + str(i)
            start = curr_bin
            i += 1
        return file_name + '_' + str(i)

    def dyn_floored_digitize(self, block_size, bin_dict, fx_names_dict, file_name):
        # floor the block size to the nearest one from the bin_dict for that particular file
        # name
        if file_name not in bin_dict:
            raise Exception("Rider Partition not the same!")
        fx_name = ""
        log_bins = bin_dict[file_name]
        start = log_bins[0]
        i = 0
        assigned = False
        for curr_bin in log_bins[1:]:
            if start <= block_size < curr_bin:
                fx_name = file_name + '_' + str(i)
                assigned = True
                break
            start = curr_bin
            i += 1
        if not assigned:
            fx_name = file_name + '_' + str(i)

        fx_name_values = fx_names_dict[file_name]
        # check if the assigned bin in assigned values of the original feature space,
        # if NOT, floor it to the nearest feature name
        if fx_name not in fx_name_values:
            index = int(fx_name.split('_')[1])
            valid_indices = sorted([int(name.split('_')[1]) for name in fx_name_values])
            start = valid_indices[0]
            assigned = False
            for curr_idx in valid_indices[1:]:
                if start <= index < curr_idx:
                    fx_name = file_name + '_' + str(index)
                    assigned = True
                    break
                start = curr_idx
            if not assigned:
                fx_name = file_name + '_' + str(valid_indices[-1])
        return fx_name

    def compute_pure_rider_block_features(self, rider_dir, attrs=['wgt']):
        for file_name in sorted(os.listdir(rider_dir)):
            if file_name == ".DS_Store":
                continue
            blocks_with_sizes = []

            for line in open(os.path.join(rider_dir, file_name)):
                line = line.strip().split()
                blocks_with_sizes.append((len(line), line))

            sorted_blocks = sorted(blocks_with_sizes, key=lambda x: x[0])

            for i, (l, block) in enumerate(sorted_blocks):
                block = set([int(n) for n in block])
                fx_name_base = file_name + '_' + str(i)
                for vertex in self.graph.nodes():
                    in_neighbours = self.graph.predecessors(vertex)
                    out_neighbours = self.graph.successors(vertex)

                    in_connections_to_block = set(in_neighbours) & block
                    in_connections_to_block_size = len(in_connections_to_block)
                    out_connections_to_block = set(out_neighbours) & block
                    out_connections_to_block_size = len(out_connections_to_block)

                    ## Local Rider Features
                    self.graph.node[vertex]['wd_' + fx_name_base] = float(
                        in_connections_to_block_size)  # destination
                    self.graph.node[vertex]['ws_' + fx_name_base] = float(
                        out_connections_to_block_size)  # source

                    for attr in attrs:
                        self.graph.node[vertex]['wda-' + attr + '_' + fx_name_base] = 0.0
                        self.graph.node[vertex]['wsa-' + attr + '_' + fx_name_base] = 0.0

                    if in_connections_to_block_size > 0:
                        for attr in attrs:
                            for connection in in_connections_to_block:
                                if attr == 'wgt':
                                    self.graph.node[vertex]['wda-' + attr + '_' +
                                                            fx_name_base] \
                                        += self.graph[connection][vertex]['weight']

                    if out_connections_to_block_size > 0:
                        for attr in attrs:
                            for connection in out_connections_to_block:
                                if attr == 'wgt':
                                    self.graph.node[vertex]['wsa-' + attr + '_' +
                                                            fx_name_base] \
                                        += self.graph[vertex][connection]['weight']
            print('RIDeR File: %s' % file_name)

    def eep_binned_block_features(self, file_name, attrs=['wgt'], bins=15):
        block_sizes = []
        block_fx_name = {}  # block_id -> feature_name
        node_block = {}  # node -> block_id in the current rider

        for line in open(file_name):
            line = line.strip().split()
            block_sizes.append(len(line))

        log_bins = np.logspace(np.log10(min(block_sizes) + 1), np.log10(max(block_sizes) + 1),
                               bins)

        for i, line in enumerate(open(file_name)):
            line = line.strip().split()
            block_size = len(line) + 1
            block_fx_name[i] = self.digitize(block_size, log_bins, file_name)

            block = set([int(n) for n in line])
            for n in block:
                n = int(n)
                node_block[n] = i
        fx_names = list(set(block_fx_name.values()))

        for vertex in self.graph.nodes():
            in_neighbours = self.graph.predecessors(vertex)
            out_neighbours = self.graph.successors(vertex)

            for fx_name_base in fx_names:
                self.graph.node[vertex]['wd-' + fx_name_base] = 0.0  # destination
                self.graph.node[vertex]['ws-' + fx_name_base] = 0.0  # source
                for attr in attrs:
                    self.graph.node[vertex]['wda-' + attr + '-' + fx_name_base] = 0.0
                    self.graph.node[vertex]['wsa-' + attr + '-' + fx_name_base] = 0.0

            for connection in in_neighbours:
                fx_name_to_be_updated = block_fx_name[node_block[connection]]
                self.graph.node[vertex]['wd-' + fx_name_to_be_updated] += 1.0
                self.graph.node[vertex]['wda-wgt-' + fx_name_base] += \
                self.graph[connection][vertex]['weight']

            for connection in out_neighbours:
                fx_name_to_be_updated = block_fx_name[node_block[connection]]
                self.graph.node[vertex]['ws-' + fx_name_to_be_updated] += 1.0
                self.graph.node[vertex]['wsa-wgt-' + fx_name_base] += \
                self.graph[vertex][connection]['weight']
        print('RIDeR File: %s' % file_name)

    # def compute_rider_binned_block_features(self, rider_dir, attrs=['wgt'], bins=15,
    # uniform=False):
    #     # Alternate log binned rider block features, binning to decrease the complexity
    #     '''fx_count = 0
    #     for file_name in sorted(os.listdir(rider_dir)):
    #         if file_name == ".DS_Store":
    #             continue
    #         if os.path.isdir(os.path.join(rider_dir, file_name)):
    #             continue
    #         block_sizes = []
    #         block_fx_name = {}  # block_id -> feature_name
    #         node_block = {}  # node -> block_id in the current rider
    #
    #         for line in open(os.path.join(rider_dir, file_name)):
    #             line = line.strip().split()
    #             block_sizes.append(len(line))
    #
    #         if not uniform:
    #             log_bins = np.logspace(np.log10(min(block_sizes)+1), np.log10(max(
    #             block_sizes)+1), bins)
    #         else:
    #             log_bins = np.linspace((min(block_sizes)+1), (max(block_sizes)+1), bins)
    #
    #         for i, line in enumerate(open(os.path.join(rider_dir, file_name))):
    #             line = line.strip().split()
    #             block_size = len(line)+1
    #             block_fx_name[i] = self.digitize(block_size, log_bins, file_name)
    #
    #             block = set([int(n) for n in line])
    #             for n in block:
    #                 n = int(n)
    #                 node_block[n] = i
    #         fx_names = list(set(block_fx_name.values()))
    #
    #         for vertex in self.graph.nodes():
    #             in_neighbours = self.graph.predecessors(vertex)
    #             out_neighbours = self.graph.successors(vertex)
    #
    #             for fx_name_base in fx_names:
    #                 self.graph.node[vertex]['wd-'+fx_name_base] = 0.0  # destination
    #                 self.graph.node[vertex]['ws-'+fx_name_base] = 0.0  # source
    #                 for attr in attrs:
    #                     self.graph.node[vertex]['wda-'+attr+'-'+fx_name_base] = 0.0
    #                     self.graph.node[vertex]['wsa-'+attr+'-'+fx_name_base] = 0.0
    #
    #             for connection in in_neighbours:
    #                 fx_name_to_be_updated = block_fx_name[node_block[connection]]
    #                 self.graph.node[vertex]['wd-'+fx_name_to_be_updated] += 1.0
    #                 self.graph.node[vertex]['wda-wgt-'+fx_name_base] += self.graph[
    #                 connection][vertex]['weight']
    #
    #             for connection in out_neighbours:
    #                 fx_name_to_be_updated = block_fx_name[node_block[connection]]
    #                 self.graph.node[vertex]['ws-'+fx_name_to_be_updated] += 1.0
    #                 self.graph.node[vertex]['wsa-wgt-'+fx_name_base] += self.graph[
    #                 vertex][connection]['weight']
    #         print('RIDeR File: %s' % file_name)'''
    #     fx_count = 0
    #     block_sizes = []
    #     block_fx_name = {}  # block_id -> feature_name
    #     node_block = {}  # node -> block_id in the current rider
    #
    #     for line in open(rider_dir):
    #         line = line.strip().split(',')
    #         block_sizes.append(len(line))
    #
    #     if not uniform:
    #         log_bins = np.logspace(np.log10(min(block_sizes)+1), np.log10(max(
    #         block_sizes)+1), bins)
    #     else:
    #         log_bins = np.linspace((min(block_sizes)+1), (max(block_sizes)+1), bins)
    #
    #     for i, line in enumerate(open(rider_dir)):
    #         line = line.strip().split()
    #         block_size = len(line)+1
    #         block_fx_name[i] = self.digitize(block_size, log_bins, rider_dir)
    #
    #         block = set([int(n) for n in line])
    #         for n in block:
    #             n = int(n)
    #             node_block[n] = i
    #     fx_names = list(set(block_fx_name.values()))
    #     print(fx_names)
    #     for vertex in self.graph.nodes():
    #         in_neighbours = self.graph.predecessors(vertex)
    #         out_neighbours = self.graph.successors(vertex)
    #
    #         for fx_name_base in fx_names:
    #             self.graph.node[vertex]['wd-'+fx_name_base] = 0.0  # destination
    #             self.graph.node[vertex]['ws-'+fx_name_base] = 0.0  # source
    #             for attr in attrs:
    #                 self.graph.node[vertex]['wda-'+attr+'-'+fx_name_base] = 0.0
    #                 self.graph.node[vertex]['wsa-'+attr+'-'+fx_name_base] = 0.0
    #
    #         for connection in in_neighbours:
    #             fx_name_to_be_updated = block_fx_name[node_block[connection]]
    #             self.graph.node[vertex]['wd-'+fx_name_to_be_updated] += 1.0
    #             self.graph.node[vertex]['wda-wgt-'+fx_name_base] += self.graph[
    #             connection][vertex]['weight']
    #
    #         for connection in out_neighbours:
    #             fx_name_to_be_updated = block_fx_name[node_block[connection]]
    #             self.graph.node[vertex]['ws-'+fx_name_to_be_updated] += 1.0
    #             self.graph.node[vertex]['wsa-wgt-'+fx_name_base] += self.graph[vertex][
    #             connection]['weight']

    def compute_rider_egonet_primitive_features(self, rider_dir, attrs=['wgt']):
        for file_name in os.listdir(rider_dir):
            print
            'RIDeR Features: ', file_name
            for i, line in enumerate(open(os.path.join(rider_dir, file_name))):
                line = line.strip().split()
                block = set([int(n) for n in line])

                neighbours = []
                for node in block:
                    neighbours.extend(self.graph.predecessors(node))
                    neighbours.extend(self.graph.successors(node))

                neighbours = list(set(neighbours))
                block_neighbours = {}
                for node in neighbours:
                    block_neighbours[node] = 1

                fx_name_base = file_name + '_' + str(i)
                for vertex in sorted(self.vertex_egonets.keys()):
                    self.init_rider_features(vertex, fx_name_base=fx_name_base, attrs=['wgt'])
                    if vertex in block_neighbours:
                        in_neighbours = self.graph.predecessors(vertex)
                        out_neighbours = self.graph.successors(vertex)

                        in_connections_to_block = set(in_neighbours) & block
                        in_connections_to_block_size = len(in_connections_to_block)
                        out_connections_to_block = set(out_neighbours) & block
                        out_connections_to_block_size = len(out_connections_to_block)

                        ## Local Rider Features
                        self.graph.node[vertex]['wd-' + fx_name_base] = float(
                            in_connections_to_block_size)  # destination
                        self.graph.node[vertex]['ws-' + fx_name_base] = float(
                            out_connections_to_block_size)  # source

                        for attr in attrs:
                            self.graph.node[vertex]['wda-' + attr + '-' + fx_name_base] = 0.0
                            self.graph.node[vertex]['wsa-' + attr + '-' + fx_name_base] = 0.0

                        if in_connections_to_block_size > 0:
                            for attr in attrs:
                                for connection in in_connections_to_block:
                                    if attr == 'wgt':
                                        self.graph.node[vertex][
                                            'wda-' + attr + '-' + fx_name_base] \
                                            += self.graph[connection][vertex]['weight']

                        if out_connections_to_block_size > 0:
                            for attr in attrs:
                                for connection in out_connections_to_block:
                                    if attr == 'wgt':
                                        self.graph.node[vertex][
                                            'wsa-' + attr + '-' + fx_name_base] \
                                            += self.graph[vertex][connection]['weight']

                for vertex in sorted(self.vertex_egonets.keys()):
                    if vertex in block_neighbours:
                        vertex_lvl_0_egonet = self.vertex_egonets[vertex][0]
                        vertex_lvl_1_egonet = self.vertex_egonets[vertex][1]

                        # Level 0 Egonet
                        for n1 in vertex_lvl_0_egonet:
                            in_neighbours = self.graph.predecessors(n1)
                            out_neighbours = self.graph.successors(n1)

                            for n2 in in_neighbours:
                                if n2 in vertex_lvl_0_egonet:
                                    self.graph.node[vertex]['wed0-' + fx_name_base] += \
                                        self.graph.node[n2]['wd-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'weda-' + attr + '0-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wda-' + attr + '-' + fx_name_base]
                                else:
                                    self.graph.node[vertex]['xed0-' + fx_name_base] += \
                                        self.graph.node[n2]['wd-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'xeda-' + attr + '0-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wda-' + attr + '-' + fx_name_base]

                            for n2 in out_neighbours:
                                if n2 in vertex_lvl_0_egonet:
                                    self.graph.node[vertex]['wes0-' + fx_name_base] += \
                                        self.graph.node[n2]['ws-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'wesa-' + attr + '0-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wsa-' + attr + '-' + fx_name_base]
                                else:
                                    self.graph.node[vertex]['xes0-' + fx_name_base] += \
                                        self.graph.node[n2]['ws-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'xesa-' + attr + '0-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wsa-' + attr + '-' + fx_name_base]

                        # Level 1 Egonet
                        for n1 in vertex_lvl_1_egonet:
                            in_neighbours = self.graph.predecessors(n1)
                            out_neighbours = self.graph.successors(n1)

                            for n2 in in_neighbours:
                                if n2 in vertex_lvl_0_egonet:
                                    self.graph.node[vertex]['wed1-' + fx_name_base] += \
                                        self.graph.node[n2]['wd-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'weda-' + attr + '1-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wda-' + attr + '-' + fx_name_base]
                                else:
                                    self.graph.node[vertex]['xed1-' + fx_name_base] += \
                                        self.graph.node[n2]['wd-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'xeda-' + attr + '1-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wda-' + attr + '-' + fx_name_base]

                            for n2 in out_neighbours:
                                if n2 in vertex_lvl_0_egonet:
                                    self.graph.node[vertex]['wes1-' + fx_name_base] += \
                                        self.graph.node[n2]['ws-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'wesa-' + attr + '1-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wsa-' + attr + '-' + fx_name_base]
                                else:
                                    self.graph.node[vertex]['xes1-' + fx_name_base] += \
                                        self.graph.node[n2]['ws-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'xesa-' + attr + '1-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wsa-' + attr + '-' + fx_name_base]

    def dyn_rider_binned_block_features(self, rider_dir, bin_dict, fx_names_dict,
                                        attrs=['wgt']):
        # Alternate log binned rider block features, binning to decrease the complexity
        fx_count = 0
        for file_name in sorted(os.listdir(rider_dir)):
            if file_name == ".DS_Store":
                continue
            block_sizes = []
            block_fx_name = {}  # block_id -> feature_name
            node_block = {}  # node -> block_id in the current rider

            for line in open(os.path.join(rider_dir, file_name)):
                line = line.strip().split()
                block_sizes.append(len(line))

            for i, line in enumerate(open(os.path.join(rider_dir, file_name))):
                line = line.strip().split()
                block_size = len(line) + 1
                block_fx_name[i] = self.dyn_floored_digitize(block_size, bin_dict,
                                                             fx_names_dict, file_name)

                block = set([int(n) for n in line])
                for n in block:
                    n = int(n)
                    node_block[n] = i
            # fx_names = list(set(block_fx_name.values()))
            # The fx_names for dyn_rider are from the riders for base network at time t.
            # print 'Derived Set diff: ', set(block_fx_name.values())
            # print 'Length of Base Set: ', set(fx_names_dict[file_name])#,
            # set(block_fx_name.values())
            fx_names = list(set(fx_names_dict[file_name]))

            for vertex in self.graph.nodes():
                in_neighbours = self.graph.predecessors(vertex)
                out_neighbours = self.graph.successors(vertex)

                for fx_name_base in fx_names:
                    self.graph.node[vertex]['wd-' + fx_name_base] = 0.0  # destination
                    self.graph.node[vertex]['ws-' + fx_name_base] = 0.0  # source
                    for attr in attrs:
                        self.graph.node[vertex]['wda-' + attr + '-' + fx_name_base] = 0.0
                        self.graph.node[vertex]['wsa-' + attr + '-' + fx_name_base] = 0.0

                for connection in in_neighbours:
                    fx_name_to_be_updated = block_fx_name[node_block[connection]]
                    if fx_name_to_be_updated not in fx_names:
                        continue
                    self.graph.node[vertex]['wd-' + fx_name_to_be_updated] += 1.0
                    self.graph.node[vertex]['wda-wgt-' + fx_name_base] += \
                    self.graph[connection][vertex]['weight']

                for connection in out_neighbours:
                    fx_name_to_be_updated = block_fx_name[node_block[connection]]
                    if fx_name_to_be_updated not in fx_names:
                        continue
                    self.graph.node[vertex]['ws-' + fx_name_to_be_updated] += 1.0
                    self.graph.node[vertex]['wsa-wgt-' + fx_name_base] += \
                    self.graph[vertex][connection]['weight']
            print('RIDeR File: %s' % file_name)

    def dyn_base_fx(self, base_rider_dir, bins=15):
        fx_names_dict = {}  # key -> file_name (partition), value -> [feature names]
        fx_bins_dict = {}  # key -> file_name, value -> [log scale]
        for file_name in sorted(os.listdir(base_rider_dir)):
            if file_name == ".DS_Store":
                continue
            block_sizes = []
            block_fx_name = {}  # block_id -> feature_name
            node_block = {}  # node -> block_id in the current rider

            for line in open(os.path.join(base_rider_dir, file_name)):
                line = line.strip().split()
                block_sizes.append(len(line))

            log_bins = np.logspace(np.log10(min(block_sizes) + 1),
                                   np.log10(max(block_sizes) + 1), bins)

            for i, line in enumerate(open(os.path.join(base_rider_dir, file_name))):
                line = line.strip().split()
                block_size = len(line) + 1
                block_fx_name[i] = self.digitize(block_size, log_bins, file_name)

                block = set([int(n) for n in line])
                for n in block:
                    n = int(n)
                    node_block[n] = i
            fx_names = list(set(block_fx_name.values()))
            fx_names_dict[file_name] = fx_names
            fx_bins_dict[file_name] = log_bins
        return fx_names_dict, fx_bins_dict

    def only_eep(self, graph_file, file_name, bins=15):
        self.load_graph(graph_file)
        self.eep_binned_block_features(file_name, attrs=['wgt'], bins=bins)
        self.no_of_vertices = self.graph.number_of_nodes()
        self.init_log_binned_fx_buckets()

        fx_names = self.get_current_fx_names()
        self.compute_log_binned_features(fx_names)

        graph_nodes = sorted(self.graph.nodes())
        fx_names = []
        for fx_name in sorted(self.graph.node[graph_nodes[0]].keys()):
            fx_names.append(fx_name)

        fx_matrix = []
        for node in graph_nodes:
            feature_row = []
            for fx_name in fx_names:
                feature_row.append(self.graph.node[node][fx_name])
            fx_matrix.append(tuple(feature_row))
        return np.array(fx_matrix)

    def only_riders(self, graph_file, rider_dir, bins=15, bin_features=True, uniform=False):
        # Compute rider features. Code is redundant, accommodates an independent riders flow
        # in riders.py
        # This will screw the graph features if used with anything else, pls refrain from
        # doing that.
        # self.load_graph(graph_file)
        if bin_features:
            if uniform:
                self.compute_rider_binned_block_features(rider_dir, attrs=['wgt'], bins=bins,
                                                         uniform=True)
            else:
                self.compute_rider_binned_block_features(rider_dir, attrs=['wgt'], bins=bins,
                                                         uniform=False)
        else:
            self.compute_pure_rider_block_features(rider_dir, attrs=['wgt'])

        self.no_of_vertices = self.graph.number_of_nodes()
        self.init_log_binned_fx_buckets()

        fx_names = self.get_current_fx_names()
        self.compute_log_binned_features(fx_names)

        graph_nodes = sorted(self.graph.nodes())
        fx_names = []
        for fx_name in sorted(self.graph.node[graph_nodes[0]].keys()):
            fx_names.append(fx_name)

        fx_matrix = []
        for node in graph_nodes:
            feature_row = []
            for fx_name in fx_names:
                feature_row.append(self.graph.node[node][fx_name])
            fx_matrix.append(tuple(feature_row))
        return np.array(fx_matrix)

    def get_current_fx_names(self):
        return [attr for attr in sorted(self.graph.nodes()[0].keys())]

    def prune_riders_fx_and_reassign_to_graph(self, riders_fx_matrix):
        pruned_riders_fx = self.prune_matrix(riders_fx_matrix, 0.0)
        curr_fx_names = self.get_current_fx_names()
        print('Current Fx: ', len(curr_fx_names))
        graph_nodes = sorted(self.graph.nodes())

        for n in graph_nodes:
            for fx in curr_fx_names:
                self.graph.node[n].pop(fx, None)

        n, f = pruned_riders_fx.shape
        for i in range(n):
            for j in range(f):
                fx_name = "fx_" + str(j)
                self.graph.node[i][fx_name] = pruned_riders_fx[i][j]

        curr_fx_names = self.get_current_fx_names()
        print('Pruned Fx: ', len(curr_fx_names))

    def only_riders_as_dict(self, graph_file, rider_dir, bins=15, bin_features=True):
        # Compute rider features. Code is redundant, accommodates an independent riders flow
        # in riders.py
        # This will screw the graph features if used with anything else, pls refrain from
        # doing that.
        # self.load_graph(graph_file)
        if not bin_features:
            self.compute_pure_rider_block_features(rider_dir, attrs=['wgt'])
        else:
            self.compute_rider_binned_block_features(rider_dir, attrs=['wgt'], bins=bins)
        self.no_of_vertices = self.graph.number_of_nodes()
        self.init_log_binned_fx_buckets()

        fx_names = self.get_current_fx_names()
        self.compute_log_binned_features(fx_names)

        graph_nodes = sorted(self.graph.nodes())
        fx_names = []
        for fx_name in sorted(self.graph.node[graph_nodes[0]].keys()):
            fx_names.append(fx_name)

        rider_features = defaultdict(list)
        for node in graph_nodes:
            feature_row = []
            for fx_name in fx_names:
                feature_row.append(self.graph.node[node][fx_name])
            rider_features[node] = feature_row
        return rider_features

    def dyn_rider(self, graph_file, num_nodes, base_rider_dir, curr_rider_dir, bins=15):
        self.load_graph_with_fixed_vertices(graph_file, num_nodes)
        fx_names_dict, fx_bins_dict = self.dyn_base_fx(base_rider_dir, bins=bins)
        self.dyn_rider_binned_block_features(curr_rider_dir, fx_bins_dict, fx_names_dict)

        self.no_of_vertices = self.graph.number_of_nodes()
        self.init_log_binned_fx_buckets()

        fx_names = self.get_current_fx_names()
        self.compute_log_binned_features(fx_names)

        graph_nodes = sorted(self.graph.nodes())
        fx_names = []
        for fx_name in sorted(self.graph.node[graph_nodes[0]].keys()):
            fx_names.append(fx_name)

        fx_matrix = []
        for node in graph_nodes:
            feature_row = []
            for fx_name in fx_names:
                feature_row.append(self.graph.node[node][fx_name])
            fx_matrix.append(tuple(feature_row))
        return np.array(fx_matrix)

    def compute_rider_egonet_primitive_features(self, rider_dir, attrs=['wgt']):
        for file_name in os.listdir(rider_dir):
            print('RIDeR Features: ', file_name)
            for i, line in enumerate(open(os.path.join(rider_dir, file_name))):
                line = line.strip().split()
                block = set([int(n) for n in line])

                neighbours = []
                for node in block:
                    neighbours.extend(self.graph.predecessors(node))
                    neighbours.extend(self.graph.successors(node))

                neighbours = list(set(neighbours))
                block_neighbours = {}
                for node in neighbours:
                    block_neighbours[node] = 1

                fx_name_base = file_name + '_' + str(i)
                for vertex in sorted(self.vertex_egonets.keys()):
                    self.init_rider_features(vertex, fx_name_base=fx_name_base, attrs=['wgt'])
                    if vertex in block_neighbours:
                        in_neighbours = self.graph.predecessors(vertex)
                        out_neighbours = self.graph.successors(vertex)

                        in_connections_to_block = set(in_neighbours) & block
                        in_connections_to_block_size = len(in_connections_to_block)
                        out_connections_to_block = set(out_neighbours) & block
                        out_connections_to_block_size = len(out_connections_to_block)

                        ## Local Rider Features
                        self.graph.node[vertex]['wd-' + fx_name_base] = float(
                            in_connections_to_block_size)  # destination
                        self.graph.node[vertex]['ws-' + fx_name_base] = float(
                            out_connections_to_block_size)  # source

                        for attr in attrs:
                            self.graph.node[vertex]['wda-' + attr + '-' + fx_name_base] = 0.0
                            self.graph.node[vertex]['wsa-' + attr + '-' + fx_name_base] = 0.0

                        if in_connections_to_block_size > 0:
                            for attr in attrs:
                                for connection in in_connections_to_block:
                                    if attr == 'wgt':
                                        self.graph.node[vertex][
                                            'wda-' + attr + '-' + fx_name_base] \
                                            += self.graph[connection][vertex]['weight']

                        if out_connections_to_block_size > 0:
                            for attr in attrs:
                                for connection in out_connections_to_block:
                                    if attr == 'wgt':
                                        self.graph.node[vertex][
                                            'wsa-' + attr + '-' + fx_name_base] \
                                            += self.graph[vertex][connection]['weight']

                for vertex in sorted(self.vertex_egonets.keys()):
                    if vertex in block_neighbours:
                        vertex_lvl_0_egonet = self.vertex_egonets[vertex][0]
                        vertex_lvl_1_egonet = self.vertex_egonets[vertex][1]

                        # Level 0 Egonet
                        for n1 in vertex_lvl_0_egonet:
                            in_neighbours = self.graph.predecessors(n1)
                            out_neighbours = self.graph.successors(n1)

                            for n2 in in_neighbours:
                                if n2 in vertex_lvl_0_egonet:
                                    self.graph.node[vertex]['wed0-' + fx_name_base] += \
                                    self.graph.node[n2]['wd-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'weda-' + attr + '0-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wda-' + attr + '-' + fx_name_base]
                                else:
                                    self.graph.node[vertex]['xed0-' + fx_name_base] += \
                                    self.graph.node[n2]['wd-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'xeda-' + attr + '0-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wda-' + attr + '-' + fx_name_base]

                            for n2 in out_neighbours:
                                if n2 in vertex_lvl_0_egonet:
                                    self.graph.node[vertex]['wes0-' + fx_name_base] += \
                                    self.graph.node[n2]['ws-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'wesa-' + attr + '0-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wsa-' + attr + '-' + fx_name_base]
                                else:
                                    self.graph.node[vertex]['xes0-' + fx_name_base] += \
                                    self.graph.node[n2]['ws-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'xesa-' + attr + '0-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wsa-' + attr + '-' + fx_name_base]

                        # Level 1 Egonet
                        for n1 in vertex_lvl_1_egonet:
                            in_neighbours = self.graph.predecessors(n1)
                            out_neighbours = self.graph.successors(n1)

                            for n2 in in_neighbours:
                                if n2 in vertex_lvl_0_egonet:
                                    self.graph.node[vertex]['wed1-' + fx_name_base] += \
                                    self.graph.node[n2]['wd-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'weda-' + attr + '1-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wda-' + attr + '-' + fx_name_base]
                                else:
                                    self.graph.node[vertex]['xed1-' + fx_name_base] += \
                                    self.graph.node[n2]['wd-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'xeda-' + attr + '1-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wda-' + attr + '-' + fx_name_base]

                            for n2 in out_neighbours:
                                if n2 in vertex_lvl_0_egonet:
                                    self.graph.node[vertex]['wes1-' + fx_name_base] += \
                                    self.graph.node[n2]['ws-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'wesa-' + attr + '1-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wsa-' + attr + '-' + fx_name_base]
                                else:
                                    self.graph.node[vertex]['xes1-' + fx_name_base] += \
                                    self.graph.node[n2]['ws-' + fx_name_base]
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            self.graph.node[vertex][
                                                'xesa-' + attr + '1-' + fx_name_base] \
                                                += self.graph.node[n2][
                                                'wsa-' + attr + '-' + fx_name_base]

    def init_vertex_egonet(self):
        for n1 in self.graph.nodes():
            if n1 not in self.vertex_egonets:
                vertex_lvl_0_egonet = self.get_egonet_members(n1)
                vertex_lvl_1_egonet = self.get_egonet_members(n1, level=1)
                self.vertex_egonets[n1] = [vertex_lvl_0_egonet, vertex_lvl_1_egonet]

    def compute_primitive_features(self, rider_fx=False, rider_dir='INVALID_PATH'):
        # computes the primitive local features
        # computes the rider based features if rider_fx is True
        # updates in place the primitive feature values with their log binned values
        for n1 in self.graph.nodes():
            if n1 not in self.vertex_egonets:
                vertex_lvl_0_egonet = self.get_egonet_members(n1)
                vertex_lvl_1_egonet = self.get_egonet_members(n1, level=1)
                self.vertex_egonets[n1] = [vertex_lvl_0_egonet, vertex_lvl_1_egonet]

            self.compute_base_egonet_primitive_features(n1, ['wgt'], level_id='0')
            self.compute_base_egonet_primitive_features(n1, ['wgt'], level_id='1')
        print('Computed Primitive Features')

        if rider_fx:
            if not os.path.exists(rider_dir):
                raise Exception("RIDeR output directory is empty!")
            # self.compute_rider_egonet_primitive_features(rider_dir, ['wgt'])
            self.compute_rider_binned_block_features(rider_dir, attrs=['wgt'], bins=15)

        self.no_of_vertices = self.graph.number_of_nodes()
        self.init_log_binned_fx_buckets()

        fx_names = self.get_current_fx_names()
        # print(fx_names)
        self.compute_log_binned_features(fx_names)

    def update_feature_matrix_to_graph(self, feature_matrix):
        # input is the structured numpy array, update these features to the graph nodes
        graph_nodes = sorted(self.graph.nodes())
        for node in graph_nodes:
            self.graph.node[node] = {}
        feature_names = list(feature_matrix.dtype.names)
        for feature in feature_names:
            values = feature_matrix[feature].tolist()
            for i, node in enumerate(graph_nodes):
                self.graph.node[node][feature] = values[i]

    def save_feature_matrix(self, dataname):
        graph_nodes = sorted(self.graph.nodes())
        feature_names = list(sorted(self.graph.node[graph_nodes[0]].keys()))
        ff = open('./rider/{}/feature-names-out.txt'.format(dataname), 'w')
        for feature in feature_names:
            ff.write('%s,' % feature)
        ff.close()

        fo = open('./rider/{}/feature-matrix.txt'.format(dataname), 'w')
        for node in graph_nodes:
            fo.write('%s' % node)
            for feature in feature_names:
                fo.write(',%s' % self.graph.node[node][feature])
            fo.write('\n')

    def compute_recursive_features(self, prev_fx_matrix, iter_no, max_dist):
        # takes the prev feature matrix and the iteration number and max_dist
        # returns the new feature matrix after pruning similar features based on the
        # similarity max dist

        print('Number of features: ', len(self.graph.node[2].keys()))
        new_fx_names = []

        for vertex in self.graph.nodes():
            new_fx_names = self.compute_recursive_egonet_features(vertex, iter_no)

            if len(new_fx_names) == 0:
                return None

        # compute and replace the new feature values with their log binned values
        self.compute_log_binned_features(new_fx_names)
        # create the feature matrix of all the features in the current graph structure
        new_fx_matrix = self.create_feature_matrix(new_fx_names)

        for name in list(prev_fx_matrix.dtype.names):
            self.memo_recursive_fx_names[name] = 1

        # return the pruned fx matrix after adding and comparing the new recursive features
        # for similarity
        return self.compare_and_prune_vertex_fx_vectors(prev_fx_matrix, new_fx_matrix,
                                                        max_dist)

    def compute_recursive_egonet_features(self, vertex, iter_no):
        # computes the sum and mean features of all the features which exist in the current
        # graph structure
        # updates the new features in place and
        # returns the string list of the new feature names

        sum_fx = '-s'
        mean_fx = '-m'
        vertex_lvl_0_egonet = self.vertex_egonets[vertex][0]
        vertex_lvl_0_egonet_size = float(len(vertex_lvl_0_egonet))
        vertex_lvl_1_egonet = self.vertex_egonets[vertex][1]
        vertex_lvl_1_egonet_size = float(len(vertex_lvl_1_egonet))

        fx_list = [fx_name for fx_name in sorted(self.graph.node[vertex].keys())
                   if fx_name not in self.memo_recursive_fx_names]
        new_fx_list = []

        level_id = '0'
        for fx_name in fx_list:
            fx_value = 0.0
            for node in vertex_lvl_0_egonet:
                fx_value += self.graph.node[node][fx_name]

            s_fx_name = fx_name + '-' + str(iter_no) + sum_fx + level_id
            m_fx_name = fx_name + '-' + str(iter_no) + mean_fx + level_id

            self.graph.node[vertex][s_fx_name] = fx_value
            self.graph.node[vertex][m_fx_name] = float(fx_value) / vertex_lvl_0_egonet_size

            new_fx_list.append(s_fx_name)
            new_fx_list.append(m_fx_name)

        # level_id = '1'
        # for fx_name in fx_list:
        #     fx_value = 0.0
        #     for node in vertex_lvl_1_egonet:
        #         fx_value += self.graph.node[node][fx_name]
        #
        #     self.graph.node[vertex][fx_name + str(iter_no) + sum_fx + level_id] = fx_value
        #     self.graph.node[vertex][fx_name + str(iter_no) + mean_fx + level_id] =
        #     fx_value / vertex_lvl_1_egonet_size
        #
        #     new_fx_list.append(fx_name + str(iter_no) + sum_fx + level_id)
        #     new_fx_list.append(fx_name + str(iter_no) + mean_fx + level_id)

        return new_fx_list

    def get_sorted_feature_values(self, feature_values):
        # takes list of tuple(vertex_id, feature value)
        # returns the sorted list using feature value as the comparison key and the length
        # of the sorted list
        sorted_fx_values = sorted(feature_values, key=lambda x: x[1])
        return sorted_fx_values, len(sorted_fx_values)

    def init_log_binned_fx_buckets(self):
        # initializes the refex_log_binned_buckets with the vertical log bin values,
        # computed based on p and the number of vertices in the graph
        max_fx_value = np.ceil(
            np.log2(self.no_of_vertices) + self.TOLERANCE)  # fixing value of p = 0.5,
        # In our experiments, we found p = 0.5 to be a sensible choice:
        # with each bin containing the bottom half of the remaining nodes.
        log_binned_fx_keys = [value for value in range(0, int(max_fx_value))]

        fx_bucket_size = []
        starting_bucket_size = self.no_of_vertices

        for idx in np.arange(0.0, max_fx_value):
            starting_bucket_size *= self.p
            fx_bucket_size.append(int(np.ceil(starting_bucket_size)))

        total_slots_in_all_buckets = sum(fx_bucket_size)
        if total_slots_in_all_buckets > self.no_of_vertices:
            fx_bucket_size[0] -= (total_slots_in_all_buckets - self.no_of_vertices)

        log_binned_buckets_dict = dict(zip(log_binned_fx_keys, fx_bucket_size))

        for binned_value in sorted(log_binned_buckets_dict.keys()):
            for count in range(0, log_binned_buckets_dict[binned_value]):
                self.refex_log_binned_buckets.append(binned_value)

        if len(self.refex_log_binned_buckets) != self.no_of_vertices:
            raise Exception("Vertical binned bucket size not equal to the number of vertices!")

    def vertical_bin(self, feature):
        # input a list of tuple(vertex_id, feature value)
        # returns a dict with key -> vertex id, value -> vertical log binned value

        vertical_binned_vertex = {}
        count_of_vertices_with_log_binned_fx_value_assigned = 0
        fx_value_of_last_vertex_assigned_to_bin = -1
        previous_binned_value = 0

        sorted_fx_values, sorted_fx_size = self.get_sorted_feature_values(feature)

        for vertex, value in sorted_fx_values:
            current_binned_value = self.refex_log_binned_buckets[
                count_of_vertices_with_log_binned_fx_value_assigned]

            # If there are ties, it may be necessary to include more than p|V| nodes
            if current_binned_value != previous_binned_value and value == \
                    fx_value_of_last_vertex_assigned_to_bin:
                vertical_binned_vertex[vertex] = previous_binned_value
            else:
                vertical_binned_vertex[vertex] = current_binned_value
                previous_binned_value = current_binned_value

            count_of_vertices_with_log_binned_fx_value_assigned += 1
            fx_value_of_last_vertex_assigned_to_bin = value

        return vertical_binned_vertex

    def compute_log_binned_features(self, fx_list):
        # input string list of feature names, which have been assigned regular feature value
        # (non-log value)
        # computes the vertical binned values for all features in the fx_list and replaces them
        # in place with their log binned values

        graph_nodes = sorted(self.graph.nodes())
        for feature in fx_list:
            node_fx_values = []
            for n in graph_nodes:
                node_fx_values.append(tuple([n, self.graph.node[n][feature]]))

            vertical_binned_vertices = self.vertical_bin(node_fx_values)
            for vertex in vertical_binned_vertices.keys():
                binned_value = vertical_binned_vertices[vertex]
                self.graph.node[vertex][feature] = float(binned_value)

    def create_initial_feature_matrix(self):
        # TODO: Code replicated between this one and the next function. Refactor.
        # Returns a numpy structured node-feature matrix for all the features assigned to
        # nodes in graph
        graph_nodes = sorted(self.graph.nodes())
        fx_names = []
        dtype = []
        for fx_name in self.get_current_fx_names():
            fx_names.append(fx_name)
            dtype.append(tuple([fx_name, '>f4']))

        fx_matrix = []
        for node in graph_nodes:
            feature_row = []
            for fx_name in fx_names:
                feature_row.append(self.graph.node[node][fx_name])
            fx_matrix.append(tuple(feature_row))
        return np.array(fx_matrix, dtype=dtype)  # return a structured numpy array

    def create_feature_matrix(self, fx_list):
        # Returns a numpy structured node-feature matrix for the features in the list
        graph_nodes = sorted(self.graph.nodes())
        fx_names = []
        dtype = []
        for fx_name in sorted(fx_list):
            fx_names.append(fx_name)
            dtype.append(tuple([fx_name, '>f4']))

        fx_matrix = []
        for node in graph_nodes:
            feature_row = []
            for fx_name in fx_names:
                feature_row.append(self.graph.node[node][fx_name])
            fx_matrix.append(tuple(feature_row))
        return np.array(fx_matrix, dtype=dtype)  # return a structured numpy array

    def fx_column_comparator(self, col_1, col_2, max_diff):
        # input two columns -> i.e. two node features and the max_dist
        # returns True/False if the two features agree within the max_dist criteria

        diff = float(max_diff) - abs(col_1 - col_2) + self.TOLERANCE
        return (diff >= self.TOLERANCE).all()

    def prune_matrix(self, actual_matrix, max_diff):
        fx_graph = nx.Graph()
        n = actual_matrix.shape[1]
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                if self.fx_column_comparator(actual_matrix[:, i], actual_matrix[:, j],
                                             max_diff):
                    fx_graph.add_edge(i, j)

        cols_to_remove = []
        connected_fx = nx.connected_components(fx_graph)
        for cc in connected_fx:
            cc = list(cc)
            for col in sorted(cc[1:]):
                cols_to_remove.append(col)

        return np.delete(actual_matrix, cols_to_remove, axis=1)

    def compare_and_prune_vertex_fx_vectors(self, prev_feature_vector, new_feature_vector,
                                            max_dist):
        # input: prev iteration node-feature matrix and current iteration node-feature
        # matrix (as structured
        # numpy array) and max_dist. Creates a feature graph based on the max_dist criteria,
        # refer the inline comments
        # on choosing the representative candidate for each connected component below.
        # Returns a numpy structured array of the final features

        fx_graph = nx.Graph()

        if prev_feature_vector is not None:
            col_prev = list(prev_feature_vector.dtype.names)
        else:
            col_prev = []

        col_new = list(new_feature_vector.dtype.names)

        # compare new features with previous features
        if len(col_prev) > 0:  # compare for something which is not a first iteration
            for col_i in col_prev:
                for col_j in col_new:
                    if self.fx_column_comparator(prev_feature_vector[col_i],
                                                 new_feature_vector[col_j], max_dist):
                        fx_graph.add_edge(col_i, col_j)

        # compare new features with new
        for col_i in col_new:
            for col_j in col_new:
                if col_i < col_j:  # to avoid redundant comparisons
                    if self.fx_column_comparator(new_feature_vector[col_i],
                                                 new_feature_vector[col_j], max_dist):
                        fx_graph.add_edge(col_i, col_j)

        # Note that a feature retained in iteration k may not be retained in iteration k + 1,
        # due to recursive features connecting them in the feature-graph.
        # In this case, we still record and output the feature because it was retained at some iteration.
        # This means that we need to keep a memo of the features

        connected_fx = nx.connected_components(fx_graph)
        for cc in connected_fx:
            old_features = []
            new_features = []

            for fx_name in cc:
                if fx_name in col_new:
                    new_features.append(fx_name)
                elif fx_name in col_prev:
                    old_features.append(fx_name)

            # Choose the representative candidate for this connected component as follows:
            #  ----------------------------------------
            # |    Old    |    New    | Representative |
            #  ----------------------------------------
            # | Empty     | Empty     | Do Nothing     |
            # | Empty     | Non empty | Pick 1 from New|
            # | Non empty | Empty     | Do Nothing     |
            # | Non empty | Non empty | Delete All New |
            #  ----------------------------------------

            if len(old_features) == 0 and len(new_features) > 0:
                features_to_be_removed = new_features[1:]
                for fx in features_to_be_removed:
                    col_new.remove(fx)  # delete all but one from new
                continue

            if len(old_features) > 0 and len(new_features) > 0:
                for fx in new_features:
                    col_new.remove(fx)  # Delete all new
                continue

        if prev_feature_vector is None:
            return new_feature_vector[col_new]

        if len(col_new) == 0:
            return prev_feature_vector

        final_new_vector = new_feature_vector[col_new]

        # return the merged fx matrix
        return merge_arrays((prev_feature_vector, final_new_vector), flatten=True)
