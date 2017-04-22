import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import signal
import cPickle
import networkx as nx
import functools
import pandas as pd
import functions as func

channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
G = nx.read_pajek("s01t01dh.net")
Gu = nx.Graph(G)
plt.figure(1)
pos = nx.random_layout(G) #fruchterman_reingold_layout, shell_layout, circular_layout, spring_layout, circular_layout, random_layout
colors = [d['weight'] for (u,v,d) in G.edges(data=True)]
nx.draw(G,pos,node_color='#93c16a',edge_color=colors,edge_cmap=plt.cm.Blues,with_labels=True, arrows = True)

A = nx.adjacency_matrix(G)

im = plt.matshow(A.todense(), cmap = plt.cm.YlOrRd, vmin=np.min(A.todense()), vmax=np.max(A.todense()))
plt.colorbar(im)

print ('no. of edges', len(G.edges()))
print ('no. of nodes', len(G.nodes()))


mat = func.normalize_data_matrix(A.todense())

c1 = 0
c2 = 0

for i in range(len(mat)):
    for j in range(len(mat)):
        if (mat[(i,j)] < 0.0 ):
            c1=c1+1
        elif(mat[(i,j)] > 1.0):
            c2=c2+1

print ('< 0.0', c1)
print ('> 1.0', c2)

print ('sum of weights', func.sum_weights(mat))
'''
print 'Out Degree -'
print func.out_degree(mat)

print 'In Degree -'
print func.in_degree(mat)

print 'Weighted Out Degree -'
print func.weighted_out_degree(mat)

print 'Weighted In Degree -'
print func.weighted_in_degree(mat)

'''
path = func.shortest_path_length(G)
weighted_path = func.shortest_weighted_path_length(G)
print ('shortest path length', path['FC1']['P3'])
print ('shortest weighted path length', weighted_path['FC1']['P3'])

print ('number of triangles in directed graph', func.triangles_directed_graph(mat))

print 'triangles directed graph'
print func.triangles_directed_per_node(mat)
print 'triangles directed graph weighted'
print func.triangles_directed_per_node_weighted(mat)

print ('weighted characteristic path length', func.weighted_characteristic_path_length(G))
print ('directed characteristic path length', func.directed_characteristic_path_length(G))

print ('directed global efficiency', func.global_efficiency_directed(G))
print ('weighted global efficiency', func.global_efficiency_weighted(G))

print ('directed clustering coefficient', func.clustering_coefficient(mat))
print ('weighted clustering coefficient', func.clustering_coefficient_weighted(Gu))

print ('weighted transitivity', func.weighted_transitivity(Gu, mat))
print ('directed transitivity', func.directed_transitivity(mat))

#print ('weighted local efficiency', func.weighted_local_efficiency(G, Gu, mat)) # 14.796752718115115 TAKES TOO MUCH TIME COZ OF SHORTEST PATH CALCULATION
print ('directed local efficiency', func.directed_local_efficiency(G, Gu, mat)) #with weight = 2.5209672703684949, without = 1.0









#plt.show()




