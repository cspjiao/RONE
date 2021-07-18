import sys
import numpy as np, networkx as nx
import random
import scipy.sparse
import collections

# <Project name: LSH>

class Graph():

	def __init__(self, adj_matrix = None, edge_time_dict = None, num_nodes = 0, num_edges = 0, start_time = None, end_time = None, 
		check_eq = True, nodes_to_explore = None, num_buckets = 0, bucket_max_value = 0, neighbor_list_static = None, neighbor_list_dynamic = None, time_edge_dict = None, 
		cat_dict = None, id_cat_dict = None, dist_scope = 0):
	
		self.adj_matrix = adj_matrix
		self.edge_time_dict = edge_time_dict
		self.num_nodes = num_nodes
		self.num_edges = num_edges
		self.start_time = start_time
		self.end_time = end_time
		self.check_eq = check_eq
		self.nodes_to_explore = nodes_to_explore

		self.num_buckets = num_buckets
		self.bucket_max_value = bucket_max_value

		self.neighbor_list_static = neighbor_list_static
		self.neighbor_list_dynamic = neighbor_list_dynamic
		self.cat_dict = cat_dict
		self.id_cat_dict = id_cat_dict

		self.time_edge_dict = time_edge_dict
		self.dist_scope = dist_scope


	def simulate_walks(self, walks_num, walk_length, nodes_to_explore, walk_mod, graph_mod, init_mod):
		'''
		Repeatedly simulate random walks from each node.
		'''
		dist_scope = self.dist_scope
		walks = []
		S_out = [ {i:[] for i in range(dist_scope)} for _ in nodes_to_explore ]
		'''
		Return (nested dicts): 
			node - {1:[nodes], 2:[nodes], 3:[nodes]}
		'''

		nodes_uniq = set([])
		nodes_all = set(nodes_to_explore)
		

		if graph_mod == 'static':
			RW_instance = RW(self, dist_scope=dist_scope)

			RW_instance.preprocess_transition_probs()

			for walk in range(walks_num):
				print(str(walk+1), '/', str(walks_num))

				for node in nodes_to_explore:
					walks.append( RW_instance.random_walk_static(walk_length=walk_length, start_node=node, walk_mod='random', S_out=S_out) )


		elif graph_mod == 'dynamic':
			RW_instance = RW(self, by_node = True, by_edge=False, dist_scope=dist_scope)

			if RW_instance.by_node:
				print('Starting random walk by nodes.')
				for walk in range(walks_num):
					print(str(walk+1), '/', str(walks_num))

					for node in nodes_to_explore:
						walks.append( RW_instance.random_walk_dynamic(walk_length=walk_length, start_node=node, walk_mod=walk_mod, init_mod= init_mod, S_out=S_out) )

			elif RW_instance.by_edge:
				print('Starting random walk by edges.')

				srcs_init = edges_sample(self.time_edge_dict, self.num_edges / 10, init_mod)

				for walk in range(walks_num):
					print(str(walk+1), '/', str(walks_num))

					for node in srcs_init:
						walk = RW_instance.random_walk_dynamic(walk_length=walk_length, start_node=node, walk_mod=walk_mod, init_mod= init_mod, S_out=S_out)
						walks.append( walk )

						for node in walk:
							nodes_uniq.add(node)

				nodes_diff = nodes_all - nodes_uniq
				print('nodes_diff:', len(nodes_diff))
				if len(nodes_diff) != 0:
					for walk in range(walks_num):
						print(str(walk+1), '/', str(walks_num))

						for node in nodes_diff:
							walk = RW_instance.random_walk_dynamic(walk_length=walk_length, start_node=node, walk_mod=walk_mod, init_mod= init_mod, S_out=S_out)
							walks.append( walk )





			else: # This should never happen
				sys.exit('[RW mode not supported]')




		else: # This should never happen
			sys.exit('[Graph mode not supported]')

			
		# print '??>>>'
		# print S_out
		return walks, S_out


class Static_Methods():

	def __init__(self, adj_matrix = None, nodes_to_explore = None):

		self.adj_matrix = adj_matrix
		self.nodes_to_explore = nodes_to_explore

	def construct_neighbor_list(self):

		adj_matrix = self.adj_matrix
		nodes_to_explore = self.nodes_to_explore

		result = {}

		# N = adj_matrix.shape[0]

		for i in nodes_to_explore:
			result[i] = list(adj_matrix.getrow(i).nonzero()[1])
			# result_r[i] = list(adj_matrix.getcol(i).nonzero()[0])
		# print result
		return result







class Dynamic_Methods():

	def __init__(self, adj_matrix = None, nodes_to_explore = None, edge_time_dict = None):

		self.adj_matrix = adj_matrix
		self.nodes_to_explore = nodes_to_explore
		self.edge_time_dict = edge_time_dict

	##############################################################################
	# Temporal walk: [by default] Walk to the closest destination node in time.
	##############################################################################

	def construct_neighbor_list(self):
		
		adj_matrix = self.adj_matrix
		nodes_to_explore = self.nodes_to_explore
		edge_time_dict = self.edge_time_dict

		if edge_time_dict is None:
			return None
			
		result = collections.defaultdict(list)
		# print edge_time_dict

		for edge in edge_time_dict:
			src = edge[0]
			dst = edge[1]
			wei = edge[2]

			occur_times = edge_time_dict[edge]

			for occur_time in occur_times:
				tup = (dst, wei, occur_time)

				result[src].append(tup)

		for edge in result:
			result[edge].sort(key=lambda x: x[2])	# ensures that the "src:(dst, wei, time)"" list is sorted
		# print result


		return result


class RW():

	def __init__(self, graph = None, by_node = False, by_edge = True, dist_scope = 0):#None, weighted = False, directed = False, neighbor_list = None, check_eq = True, alias_nodes = None):
		self.graph = graph
		self.by_node = by_node
		self.by_edge = by_edge
		self.dist_scope = dist_scope

		np.seterr(all='raise')


	#########################################################################################
	# Static RW
	#########################################################################################


	def preprocess_transition_probs(self):
		adj_matrix = self.graph.adj_matrix
		neighbor_list = self.graph.neighbor_list_static

		# <TODO>: add cases about directed / weighted

		alias_nodes = {}
		nodes = neighbor_list.keys()

		# print neighbor_list

		for node in nodes:
			unnormalized_probs = [adj_matrix[node, nbr] for nbr in sorted(neighbor_list[node])]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		self.alias_nodes = alias_nodes

		return



	def random_walk_static(self, walk_length, start_node, walk_mod, S_out):

		adj_matrix = self.graph.adj_matrix
		neighbor_list = self.graph.neighbor_list_static
		alias_nodes = self.alias_nodes
		dist_scope = self.dist_scope

		walk = [start_node]
		# walk_role = [start_node]
		# walk_role = [roles_mapping[start_node]]

		while len(walk) < walk_length:
			cur_node = walk[-1]
			cur_node_neighbors = sorted(neighbor_list[cur_node])
			if len(cur_node_neighbors) > 0:

				if walk_mod == 'random':
					next_node = cur_node_neighbors[alias_draw(alias_nodes[cur_node][0], alias_nodes[cur_node][1])]

					# print next_node
					# print walk

					for idx in range( min(len(walk),  dist_scope) ):

						# print S_out
						S_out[walk[-1-idx]][idx].append(next_node)


					walk.append(next_node)
					# walk_role.append(roles_mapping[next_node])

				else:
					sys.exit('error: random walk mode not supported.')
			else:
				break

		return walk


	#########################################################################################
	# Dynamic RW
	#########################################################################################


	def sample_next(self, neighbors_all, duration, cur_time, strategy):	# neighbors_all: [(0, 1.0, 2), (3, 1.0, 2), (7, 5.0, 2)]
		normalizer = 0
		denormalizers = []
		p = []

		if len(neighbors_all) == 0:
			return []

		start_time = neighbors_all[0][2]
		end_time = neighbors_all[-1][2]

		if (strategy == 'random') or (start_time == end_time):
			for dst, wei, time in neighbors_all:

				denormalizer = 1.0
				normalizer += denormalizer

				denormalizers.append(denormalizer)

		elif strategy == 'late':
			for dst, wei, time in neighbors_all:

				denormalizer = round(np.exp(normalize(time, cur_time, duration)), 3)
				normalizer += denormalizer

				denormalizers.append(denormalizer)
	            
		elif strategy == 'early':
			# print 'Current_iteration'
			for dst, wei, time in neighbors_all:

				try:
					denormalizer = round(np.exp(normalize(cur_time, time, duration)), 3)

				except:
					print(cur_time)
					print(dst, wei, time)
					print('------')
					print(str(end_time-start_time))
					print(normalize(time, cur_time, duration))
					sys.exit(':(')
				normalizer += denormalizer

				denormalizers.append(denormalizer)
	            
		else:
			sys.exit('error: random walk mode not supported.')


		p = [ele/normalizer for ele in denormalizers]
		
		# print p
		# print '-------'
		# print neighbors_all[:100]
		
		try:
			next_dst_wei_time_idx = np.random.choice(range(len(neighbors_all)), 1, p=p)[0]
		except:
			print(p)
			sys.exit('Something went wrong.')
		next_dst_wei_time = neighbors_all[next_dst_wei_time_idx]

		return next_dst_wei_time



	def random_walk_dynamic(self, walk_length, start_node, walk_mod, init_mod, S_out):


		adj_matrix = self.graph.adj_matrix
		neighbor_list_dynamic = self.graph.neighbor_list_dynamic
		edge_time_dict = self.graph.edge_time_dict
		start_time_g = self.graph.start_time
		end_time_g = self.graph.end_time
		duration = end_time_g - start_time_g
		dist_scope = self.dist_scope

		# print start_node
		# print '----'
		walk = [start_node]
		# print 'start_node: ' + str(start_node)

		#self.sample_init_edge(neighbor_list_dynamic, start_node, walk_length, mod)

		
		init_dst_wei_time = self.sample_next(neighbor_list_dynamic[start_node], duration, start_time_g, init_mod)

		# print '======='
		# print start_node
		# print neighbor_list_dynamic
		# print init_dst_wei_time
		if len(init_dst_wei_time) == 0:
			return walk

		cur_time = init_dst_wei_time[2]
		walk.append(init_dst_wei_time[0])


		while len(walk) < walk_length:
			cur_node = walk[-1]

			try:
				cur_idx = next( x[0] for x in enumerate(neighbor_list_dynamic[cur_node]) if x[1][2] > cur_time )
				# print 'cur_idx: ' + str(cur_idx)
			except:
				break

			neighbors_all = [ ele for ele in neighbor_list_dynamic[cur_node] ][cur_idx:]

			if len(neighbors_all) > 0:

				next_dst_wei_time = self.sample_next(neighbors_all, duration, cur_time, walk_mod)
				next_node = next_dst_wei_time[0]

				for idx in range( min(len(walk),  dist_scope) ):
					S_out[walk[-1-idx]][idx].append(next_node)

				walk.append(next_node)
				
				# print 'cur_time:' + str(cur_time)
			else:
				break

			# print 'current_walk' + str(walk)
			cur_time = max(next_dst_wei_time[2], cur_time)
			# print 'updated_time:' + str(cur_time)

		# print '??'
		# print walk
		return walk
		# return walk_role

		




########################################################
# Common tools to use
########################################################

def get_delimiter(input_file_path):
	delimiter = " "
	if ".csv" in input_file_path:
		delimiter = ","
	elif ".tsv" in input_file_path:
		delimiter = "\t"
	else:
		print(input_file_path)
		sys.exit('Format not supported.')

	return delimiter

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]

def normalize(x, x_cur, duration):
	return float(x-x_cur)/float(duration)


def edges_sample(time_edge_dict, c, init_mod):
	time_min = min( time_edge_dict.keys() )
	time_max = max( time_edge_dict.keys() )
	time_range = time_max - time_min

	if init_mod == 'early':

		pivot = time_min + time_range * 1 / 4

		candidates_raw_l = pivot - time_range * 1 / 2
		candidates_raw_u = pivot + time_range * 1 / 2

		candidates_raw_list = list()

		for time in time_edge_dict:
			if time >= candidates_raw_l and time <= candidates_raw_u:
				candidates_raw_list.append(time)

		candidates = np.random.choice(candidates_raw_list, c, replace=True)

	elif init_mod == 'late':

		pivot = time_min + time_range * 3 / 4

		candidates_raw_l = pivot - time_range * 1 / 2
		candidates_raw_u = pivot + time_range * 1 / 2

		candidates_raw_list = list()

		for time in time_edge_dict:
			if time >= candidates_raw_l and time <= candidates_raw_u:
				candidates_raw_list.append(time)

		candidates = np.random.choice(candidates_raw_list, c, replace=True)

	elif init_mod == 'random':

		candidates = np.random.choice(time_edge_dict.keys(), c, replace=True)

	else:
		sys.exit('init_mode does not supported.')


	srcs = list()

	for time in candidates:

		candidates_num = len(time_edge_dict[time])

		if candidates_num > 1:
			idx_random = np.random.choice(range(candidates_num), 1)[0]
			src = time_edge_dict[time][idx_random][0]

		else:
			src = time_edge_dict[time][0][0]

		srcs.append(src)
	# print '========='
	# print srcs
	return srcs



def write_embedding(rep, output_file_path):
	N, K = rep.shape

	fOut = open(output_file_path, 'w')
	fOut.write(str(N) + ' ' + str(K) + '\n')

	for i in range(N):
		cur_line = ' '.join([str(np.round(ii, 6)) for ii in rep[i,:]])
		fOut.write(str(i) + ' ' + cur_line + '\n')

	fOut.close()

	return


