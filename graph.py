import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import signal
import cPickle
import networkx as nx

def draw(ls):
    G = nx.DiGraph()

    channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

    #G.add_nodes_from(channels)

    #G.add_nodes_from(ls)
    loclist=[]
    for l in ls:
        local=[]
        for k in l:
            if isinstance(k,int):
                local.append(k)
                #local.append(channels[k])
            else:
                local.append(k)
        loclist.append(local)



    G.add_weighted_edges_from(loclist)

    #print (G.edges())

    # spring graphy layout
    plt.figure(1)
    pos = nx.random_layout(G) #fruchterman_reingold_layout, shell_layout, circular_layout, spring_layout, circular_layout, random_layout
    colors = [d['weight'] for (u,v,d) in G.edges(data=True)]
    #nx.write_pajek(G, "s01t01ua.net")
    nx.draw(G,pos,node_color='#A0CBE2',edge_color=colors,edge_cmap=plt.cm.Blues,with_labels=True)

    #A = nx.adjacency_matrix(G)
    #plt.matshow(A.todense(),cmap=plt.cm.Blues)
    plt.show()