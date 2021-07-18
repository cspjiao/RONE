import numpy as np

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def genPosNegPairs(similarity, k):
	trainParis = []
	numOfNodes = len(similarity)
	for i in range(numOfNodes):
		sortedIndices = argsort(similarity[i])
		for j in range(k):
			pair = []
			pair.append(i)
			pair.append(sortedIndices[-2-j])
			pair.append(i)
			pair.append(sortedIndices[j])
			pair.append(0)
			trainParis.append(pair)
	return trainParis
