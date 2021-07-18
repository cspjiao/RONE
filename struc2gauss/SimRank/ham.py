import calSimRank
import GenPosNegPairsbyRandom
import networkx as nx
import numpy as np

fh = open("/home/ypei1/Embedding/NewData/CA-GrQc.edgelist", 'rb')
G = nx.read_edgelist(fh, nodetype=int)
fh.close()

numOfNodes = len(G.nodes())

fout = open('/home/ypei1/Embedding/NewData/CA-GrQc.simrank.sim', 'w')

simDict = calSimRank.simrank(G)
similarity = [[0.0 for i in range(numOfNodes)] for j in range(numOfNodes)]
for key, val in simDict.items():
	for k, v in val.items():
		similarity[key][k] = v
		fout.write(str(v)+' ')
	fout.write('\n')

fout.close()

#training_data = np.asarray(GenPosNegPairsbyRandom.genPosNegPairs(similarity, 5, 1), dtype=np.uint32)
#print training_data

#print training_data[:,4]