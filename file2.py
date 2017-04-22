import networkx as nx
import graphviz

import numpy as np

numBins = 10  # number of bins in each dimension
data = np.random.randn(100000, 3)  # generate 100000 3-d random data points
jointProbs, edges = np.histogramdd(data, bins=numBins)
jointProbs /= jointProbs.sum()

print type(data)
print data.shape
print type(jointProbs)
print jointProbs.shape
print type(edges)

