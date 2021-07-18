from scipy.optimize import linear_sum_assignment as lsa
import numpy as np

import networkx as nx
import time
import sys


def maximal_matching(G, similarities, n1, n2):
	cost = np.empty((len(n1), len(n2)))

	for i, v in enumerate(n1):
		for j, w in enumerate(n2):
			if (v,w) in similarities:
				cost[i, j] = 1 - similarities[v, w]
			else:
				# estimate similarities for non-candidate pair
				cost[i, j] = 1 - alpha * degree_ratio(G.degree[v], G.degree[w])

	row_ind, col_ind = lsa(cost)
	return np.array([1 - x for x in cost[row_ind, col_ind]]).sum()


def degree_ratio(du, dv):
	return (1 - beta) * (min(du, dv) / max(du, dv)) + beta


def epoch(G, similarities):
	new_sims = {}
	done = True
	for (u, v), sim in similarities.items():
		if u == v:
			new_sims[u, v] = 1.0
			continue

		du = G.degree[u]
		nu = [n for n in G[u]]

		dv = G.degree[v]
		nv = [n for n in G[v]]

		match_size = min(len(nu), len(nv))
		match_similarity = maximal_matching(G, similarities, nu, nv)
		new_sims[u, v] = (1.0 - beta) * (match_similarity / (du + dv - match_size)) + beta
		if abs(new_sims[u, v] - similarities[u, v]) > delta:
			done = False
	return done, new_sims


def initialize(G):
	sims = {}
	neighs = {}
	verts = set()
	vertices = sorted(G, key=lambda x: G.degree[x])
	for v in G:
		neighs[v] = sorted([x[1] for x in G.degree(G[v])])

	#print(neighs[1])
	for u in vertices:
		for v in vertices:
			du = G.degree[u]
			dv = G.degree[v]

			if du == 0 or dv == 0:
				continue
			if not (theta_pr * du <= dv and dv <= du):
				continue

			du1 = neighs[u][0]
			dv1 = neighs[v][0]
			dr = degree_ratio(du1, dv1)

			if dv1 <= du1 and dv - 1 + dr < theta_pr * du:
				continue

			temp_sims = {}
			for x in G[u]:
				for y in G[v]:
					temp_sims[x, y] = degree_ratio(G.degree[x], G.degree[y])
			w = maximal_matching(G, temp_sims, G[u], G[v])

			if w >= theta_pr * du:
				sims[u, v] = (1 - beta) * w / du + beta
				verts.add(u)
				verts.add(v)

	return verts, sims


def load_edges(filename):
	edges = []
	with open(filename) as f:
		for line in f:
			tmp = line.strip().split()
			edges.append([int(tmp[0]), int(tmp[1])])
	return edges


if __name__ == '__main__':
	alpha = 0.5
	beta = 0.1
	delta = 0.01
	theta = 0.9
	theta_pr = (theta - beta) / (1 - beta)

	start_time = time.time()

	'''
	A = np.matrix(np.loadtxt('epinions.txt'))
	G = nx.from_numpy_matrix(A)
	'''

	number_of_nodes = 12645
	G = nx.Graph()
	edges = load_edges('/home/ypei1/Embedding/RoleSim/graph/anybeat.edgelist')

	G.add_nodes_from([i for i in range(number_of_nodes)])
	G.add_edges_from(edges)

	print(len(G), G.size())

	verts, sims = initialize(G)
	done = False
	iter = 10
	#while not done:
	for i in range(5):
		#iter += 1
		done, sims = epoch(G, sims)

	elapsed_time = time.time() - start_time

	print('elapsed time: {}'.format(elapsed_time))
	#print('iteratioin time: {}'.format(iter))

	#number_of_nodes = len(G)
	similarity = [[0.0 for i in range(number_of_nodes)] for j in range(number_of_nodes)]
	for k, v in sims.items():
		(from_id, to_id) = k
		similarity[from_id][to_id] = v
		similarity[to_id][from_id] = v

	np.savetxt('anybeat.ice.sim', np.asarray(similarity))
