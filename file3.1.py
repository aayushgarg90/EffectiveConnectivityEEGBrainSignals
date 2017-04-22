import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import signal
import cPickle
import networkx as nx
import functools
import pandas as pd
import functions as func


outfile = 'Schreiber_matrix.npy'
thres = 0.0


raw1 = cPickle.load(open('data_preprocessed_python/s01.dat', 'rb'))

print(raw1['data'].shape)
print (raw1['labels'].shape)

raw2 = raw1['data']

data = raw2[0]
print data.shape
print type(data)
labels = raw1['labels']
print 'labels'


x1 = np.linspace(0, 8063, num=8064)
x2 = x1
print ('x.size', x1.size)

'''
mat = np.zeros((32, 32))


#Schreibers transfer entropy
mat = np.zeros((32, 32))

for i in range(len(data)-8):
    for j in range(len(data)-8):
        y1 = data[i]
        y2 = data[j]

        y1[y1 <= 0.0] = 0
        y1[y1 > 0.0] = 1
        y2[y2 <= 0.0] = 0
        y2[y2 > 0.0] = 1

        X = [int(x) for x in y1]
        Y = [int(x) for x in y2]

        #print ('sum y1', sum(X))
        #print ('sum y2', sum(Y))
        #X = (1,1,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,0,1,1,0,1)
        #Y = (1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,1)
        val1 = func.SchreiberEntropy(X,Y)

        mat[i][j] = val1

        #print ('X,Y', func.SchreiberEntropy(X,Y)) # 0.161196007927
        #print ('Y,X', func.SchreiberEntropy(Y,X)) # 0.470027584474


np.save(outfile, mat)

'''
mat = np.load(outfile)
mat1 = mat

for i in range(len(data)-8):
    for j in range(len(data)-8):
        mat1[i][j] = (-1) * mat[j][i]

G = nx.DiGraph()

channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

G.add_nodes_from(channels)

edges = []


for i in range(len(data)-8):
    for j in range(len(data)-8):
        mat[i][j] = mat[i][j] * 100
        if (mat[i][j] < 0.0):
            mat[i][j] = mat[i][j] + 11.00
        elif (mat[i][j] > 0.0):
            mat[i][j] = mat[i][j] - 11.00


#mat = func.normalize_data(mat)

for i in range(len(data)-8):
    for j in range(len(data)-8):
        if (mat1[i][j] >= thres and i != j):
            edges.append((channels[i], channels[j], mat[i][j]))


print ('no. of edges', len(edges))
G.add_weighted_edges_from(edges)
print (G.edges())

# spring graphy layout
plt.figure(1)
pos = nx.random_layout(G) #fruchterman_reingold_layout, shell_layout, circular_layout, spring_layout, circular_layout, random_layout
colors = [d['weight'] for (u,v,d) in G.edges(data=True)]
#nx.write_pajek(G, "s01t01da.net")
#G=nx.read_pajek("test.net")
nx.draw(G,pos,node_color='#93c16a',edge_color=colors,edge_cmap=plt.cm.Blues,with_labels=True, arrows = True)

A = nx.adjacency_matrix(G)

im = plt.matshow(A.todense(), cmap = plt.cm.YlOrRd, vmin = np.min(mat), vmax = np.max(mat))
#plt.figure(2)
#im = plt.imshow(A.todense(), vmin=np.min(mat), interpolation='none',vmax=np.max(mat), cmap=plt.cm.YlOrRd, aspect='auto')
plt.colorbar(im)

nm = np.zeros((32,32))

for i in range(len(data)-8):
    for j in range(len(data)-8):
        if (mat1[i][j] >= thres and i != j):
            nm[i][j] = 1

fl = 0
for i in range(len(data)-8):
    for j in range(len(data)-8):
        if (nm[i][j] == 1 and nm[j][i] == 1 and i != j):
            fl = fl + 1

print ('number of bidirectional connections', fl)
#print A.todense()
plt.show()
