import networkx as nx
import collections
import copy

def simrank(G, r=0.9, max_iter=100):
	# init. vars
	sim_old = collections.defaultdict(list)
	sim = collections.defaultdict(list)
	for n in G.nodes():
		sim[n] = collections.defaultdict(int)
		sim[n][n] = 1
		sim_old[n] = collections.defaultdict(int)
		sim_old[n][n] = 0

	# recursively calculate simrank
	for iter_ctr in range(max_iter):
		if _is_converge(sim, sim_old):
			break
		sim_old = copy.deepcopy(sim)
		for u in G.nodes():
			for v in G.nodes():
				if u == v:
					continue
				s_uv = 0.0
				for n_u in G.neighbors(u):
					for n_v in G.neighbors(v):
						s_uv += sim_old[n_u][n_v]
				sim[u][v] = (r * s_uv / (len(G.neighbors(u)) * len(G.neighbors(v))))
	return sim

def _is_converge(s1, s2, eps=1e-4):
	for i in s1.keys():
		for j in s1[i].keys():
			if abs(s1[i][j] - s2[i][j]) >= eps:
				return False
	return True
