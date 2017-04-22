import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import signal
import cPickle
import networkx as nx
import functions as func


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
print x1.size
plv = []
mat = np.zeros((32, 32))

for i in range(len(data)-8):
    for j in range(len(data)-8):

        y1 = data[i]#[:4032]
        y2 = data[j]#[:4032]

        phase1 = np.angle(y1,deg=False)
        phase2 = np.angle(y2,deg=False)

        phase_diff = phase1 - phase2
        phase_diff2 = phase2 - phase1

        complex_phase_diff = np.exp(np.complex(0, 1)*(phase_diff))
        complex_phase_diff2 = np.exp(np.complex(0, 1) * (phase_diff2))

        #print (np.sum(complex_phase_diff), np.sum(complex_phase_diff2))
        #print (np.absolute(np.sum(complex_phase_diff)), np.absolute(np.sum(complex_phase_diff2)))

        plv1 = np.abs(np.sum(complex_phase_diff))/len(phase1)
        plv.append(plv1)
        mat[i][j] = plv1

thres = func.calculate_plv_threshold(data)

print ('plv thres', thres)

G = nx.Graph()

channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

G.add_nodes_from(channels)

edges = []
cnt = 0

for i in range(len(data)-8):
    for j in range(len(data)-8):
        if (mat[i][j] >= thres and i != j):
            cnt = cnt + 1
            edges.append((channels[i], channels[j], mat[i][j]))


print ('no. of edges', len(edges), cnt)
G.add_weighted_edges_from(edges)
print (G.edges())

# spring graphy layout
plt.figure(1)
pos = nx.random_layout(G) #fruchterman_reingold_layout, shell_layout, circular_layout, spring_layout, circular_layout, random_layout
colors = [d['weight'] for (u,v,d) in G.edges(data=True)]
#nx.write_pajek(G, "s01t01ua.net")
nx.draw(G,pos,node_color='#A0CBE2',edge_color=colors,edge_cmap=plt.cm.Blues,with_labels=True)

A = nx.adjacency_matrix(G)
plt.matshow(A.todense(),cmap=plt.cm.Blues)

fl=0
for i in range(len(data)-8):
    for j in range(len(data)-8):
        if (mat[i][j] != mat[j][i] and i != j):
            fl = 1
            break
    if (fl == 1):
        break

if (fl == 1):
    print "not symmetrical"

plt.show()













