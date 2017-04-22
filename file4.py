import random
import networkx as nx
from networkx.algorithms.bipartite import biadjacency_matrix
import matplotlib.pyplot as plt
# generate random bipartite graph, part 1: nodes 0-9, part 2: nodes 10-29
B=nx.dodecahedral_graph()
# add some random weights
for u,v in B.edges():
    B[u][v]['weight']=random.randint(0,4)

# spring graphy layout
plt.figure(1)
pos = nx.spring_layout(B)
colors = [d['weight'] for (u,v,d) in B.edges(data=True)]
nx.draw(B,pos,node_color='#A0CBE2',edge_color=colors,width=4,edge_cmap=plt.cm.Blues,with_labels=False)
#plt.savefig('one.png')

'''
# simple bipartite layout
plt.figure(2)
pos = {}
for n in range(10):
    pos[n]=(n*2,1)
for n in range(10,30):
    pos[n]=(n-10,0)
nx.draw(B,pos,node_color='#A0CBE2',edge_color=colors,width=4,edge_cmap=plt.cm.Blues,with_labels=False)
#plt.savefig('two.png')
'''

# biadjacency matrix colormap
#M = biadjacency_matrix(B,row_order=range(len(B.nodes())),column_order=range(len(B.nodes())))
A = nx.adjacency_matrix(B)
print(A.todense())
plt.matshow(A.todense(),cmap=plt.cm.Blues)
#plt.savefig('three.png')
plt.show()