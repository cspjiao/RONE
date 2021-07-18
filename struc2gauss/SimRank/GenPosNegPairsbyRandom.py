import numpy as np
import random

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def genPosNegPairs(similarity, window_size, nsamples):
	trainParis = []
	numOfNodes = len(similarity)
	for i in range(numOfNodes):
		sortedIndices = argsort(similarity[i])
		postiveIndices = sortedIndices[-1-window_size:-1]
		negativeIndices = sortedIndices[:numOfNodes-window_size]
		for j in range(window_size):
			for k in range(nsamples):
				pair = []
				pair.append(i)
				pair.append(postiveIndices[j])
				pair.append(i)
				pair.append(negativeIndices[random.randint(0, numOfNodes-window_size-1)])
				pair.append(0)
				trainParis.append(pair)
	return trainParis
