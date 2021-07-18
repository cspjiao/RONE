import sys
import os
import argparse
import networkx as nx
import pickle as pkl
import pandas as pd


graph = {}
pie = {0: []}
active = {0: []}

parser = argparse.ArgumentParser(description="eer.")
parser.add_argument('--dataset', nargs = '?', default = "clf/brazil-flights", help = 'dataset')
parser.add_argument('--lpmethod', nargs='?', default='Hadamard', help='binary operator')
parser.add_argument('--epsilon', type=int, default=2, help='binary operator')
args = parser.parse_args()


if 'clf/' in args.dataset:
    input_file_path = "../dataset/" + args.dataset + ".edge"
elif 'lp/' in args.dataset:
    input_file_path = "../cache/" + args.dataset + "-1.pkl"
epsilon = args.epsilon
    
    
def create_graph():
    '''node_set = set()
    for line in open(input_file):
        # source, destination, weight
        line = line.strip()
        line = line.split(' ')
        source = int(line[0])
        dest = int(line[1])
        node_set.add(source)
        node_set.add(dest)
    num_nodes = len(node_set)
    for i in range(0, num_nodes):
        graph[i] = []

    for line in open(input_file):
        # source, destination, weight
        line = line.strip()
        line = line.split(' ')
        source = int(line[0])
        dest = int(line[1])
        #wt = int(line[2])
        graph[source].append(dest)'''
    
    #print "Graph %s populated in memory " % input_file
    '''degree = 0
    zero_nodes = 0
    for node in graph.keys():
        if len(graph[node]) == 0:
            zero_nodes += 1
        degree += len(graph[node])'''
    if "/clf/" in input_file_path:
        G = nx.from_edgelist(pd.read_csv(input_file_path, encoding='utf-8', header=-1, sep=' ').values.tolist())
        G.remove_edges_from(G.selfloop_edges())
    elif "/lp/" in input_file_path:
        datafile = open(input_file_path,'rb')
        graphAttr = pkl.load(datafile)
        G = graphAttr['G_rmd']
        datafile.close()
    for i in G.nodes():
        graph[i] = list(G.neighbors(i))
    print(graph)
    #print('num nodes: {}, zero deg nodes: {}, total degree: {}, avg degree: {}'.format(num_nodes, zero_nodes, degree, float(degree) / (num_nodes-zero_nodes)))
    
    return graph


def initialize(pie, active):
    for index in range(len(graph)):
        pie[0].append(index)
    for val in pie[0]:
        active[0].append(val)
    return


def degree_dist(active_cell):
    fofU = [0 for i in range(len(graph))]
    for index in graph.keys():
        fofU[index] = len(set(graph[index]) & set(active_cell))
    return fofU


'''def display(pie):
    fo = open(output_file, "w")
    for key in sorted(pie.keys()):
        cell = pie[key]
        for v in sorted(cell):
            fo.write(str(v) + " ")
        fo.write("\n")
    fo.close()'''


def split(cell, fofU, epsilon):
    def align_split_cells(split_cells):
        idx = 1
        aligned_split_cells = {}
        for key in sorted(split_cells.keys()):
            aligned_split_cells[idx] = []
            for v in split_cells[key]:
                aligned_split_cells[idx].append(v)
            idx += 1
        return aligned_split_cells

    split_cells = {}
    aligned_split_cells = {}

    for vertex in cell:
        key = fofU[vertex]
        if key not in split_cells:
            split_cells[key] = [vertex]
        else:
            split_cells[key].append(vertex)
    # the partition till now in splittedCells is per Equitable Partition definition

    sorted_split_cell_keys = sorted(split_cells.keys())
    if len(sorted_split_cell_keys) > 1:
        split_boundaries = {}
        boundary_index = 1
        start_key = sorted_split_cell_keys[0]
        split_boundaries[boundary_index] = [start_key]
        for i in range(1, len(sorted_split_cell_keys)):
            end_key = sorted_split_cell_keys[i]
            if (end_key - start_key) <= epsilon:
                split_boundaries[boundary_index].append(end_key)
            else:
                boundary_index += 1
                split_boundaries[boundary_index] = [end_key]
                start_key = end_key
        for k in sorted(split_boundaries.keys()):
            aligned_split_cells[k] = []
            for b in split_boundaries[k]:
                for v in split_cells[b]:
                    aligned_split_cells[k].append(v)
    else:
        aligned_split_cells = align_split_cells(split_cells)
    return aligned_split_cells


def is_cell_in_active(sorted_cell):
    for key in active.keys():
        cell_in_active = sorted(active[key])
        if cell_in_active == sorted_cell:
            return key
    return -1


graph = create_graph()

initialize(pie, active)

max_index_in_pie = 0
max_index_in_active = 0
num_nodes = len(graph)
no_iters = 0

while active and (len(pie) != num_nodes):
    no_iters += 1
    active_keys = sorted(active.keys())
    min_index = min(active_keys)
    active_cell = active[min_index]
    del (active[min_index])
    fofU = degree_dist(active_cell)
    tmp_pie_keys = list(pie.keys())
    for key in tmp_pie_keys:
        cell = pie[key]
        split_cells = split(cell, fofU, epsilon)
        split_cells_keys = split_cells.keys()
        if len(split_cells_keys) == 1:
            continue
        s = max(split_cells_keys)  # s in paper
        t = 1  # t in paper
        t_finder = {}
        for k in sorted(split_cells_keys):
            l = len(split_cells[k])
            if l > s:
                break
            else:
                if l not in t_finder:
                    t_finder[l] = [k]
                else:
                    t_finder[l].append(k)
        t_finder_keys = sorted(t_finder.keys())
        if len(t_finder_keys) > 0:
            t = t_finder[max(t_finder_keys)][0]  # Let t be tbe smallest integer such |X_t| is max(1 <= t <= s)
        else:
            t = 1
        del (pie[key])
        for s in split_cells.keys():
            max_index_in_pie += 1
            pie[max_index_in_pie] = split_cells[s]
        print(cell)
        does_cell_belong_to_active = is_cell_in_active(sorted(cell))  # check if cell in a member of active
        if does_cell_belong_to_active != -1:
            del (active[does_cell_belong_to_active])
            active[does_cell_belong_to_active] = split_cells[t]

        for i in range(1, t):
            max_index_in_active += 1
            active[max_index_in_active] = split_cells[i]
        for i in range(t + 1, s + 1):
            max_index_in_active += 1
            active[max_index_in_active] = split_cells[i]
    if no_iters % 100 == 0:
        print('No iters = %s, Size of active = %s, Size of pie = %s'.format(no_iters, len(active), len(pie)))

print(pie)
print(active)
#display(pie)

# TODO: 1. Sort Active according to size (lowest to highest) 2. Introduce FoFUDistributed and run time experiments