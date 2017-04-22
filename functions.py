import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import signal
import cPickle
import networkx as nx
import functools
import pandas as pd
from itertools import permutations

ch = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2',
      'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']


def reduce_concat(x, sep=""):
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)

def paste(lists, sep="", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)

def SchreiberEntropy(X,Y,s=1):
    L4 = L1 = len(X) - s  # Lengths of vector Xn+1.
    L3 = L2 = len(X)  # Lengths of vector Xn (and Yn).

    '''
    print ('X', X)
    print ('Y', Y)
    print ('L1', L1)
    print ('L2', L2)
    print ('L3', L3)
    print ('L4', L4)
    '''

    # -------------------#
    # 1. p(Xn+s,Xn,Yn): #
    # -------------------#

    TPvector1 = []  # np.zeros(L1) # Init.
    #print ('TPvector1', TPvector1)

    for i in range(L1):
        vect = (X[i + s], "i", X[i], "i", Y[i])
        TPvector1.append(paste(vect, collapse=""))  # "addresses"

    #print ('TPvector1', TPvector1)
    #print ('len(TPvector1)', len(TPvector1))

    ps = pd.Series([tuple(i) for i in TPvector1])
    TPvector1C = ps.value_counts()
    #print ('TPvector1C.type', type(TPvector1C))
    #print ('TPvector1C', TPvector1C)

    TPvector1T = TPvector1C / len(TPvector1)  # Table of probabilities.

    #print ('TPvector1T.type', type(TPvector1T))
    #print ('TPvector1T', TPvector1T)

    '''
    lst = []
    vect = ('1',"i",'1',"i",'1')
    lst.append(paste(vect,collapse=""))
    print lst
    print tuple(vect)
    print TPvector1T.index[0]
    print TPvector1T[tuple(vect)]

    '''

    # -----------#
    # 2. p(Xn): #
    # -----------#

    TPvector2 = X
    ps2 = pd.Series([i for i in TPvector2])
    TPvector2C = ps2.value_counts()
    TPvector2T = TPvector2C / sum(TPvector2C)
    #print ('TPvector2T.type', type(TPvector2T))
    #print ('TPvector2T', TPvector2T)

    # --------------#
    # 3. p(Xn,Yn): #
    # --------------#

    TPvector3 = []

    for i in range(L3):
        vect = (X[i], "i", Y[i])
        TPvector3.append(paste(vect, collapse=""))  # "addresses"

    #print ('TPvector3', TPvector3)
    #print ('len(TPvector3)', len(TPvector3))

    ps = pd.Series([tuple(i) for i in TPvector3])
    TPvector3C = ps.value_counts()
    #print ('TPvector3C.type', type(TPvector3C))
    #print ('TPvector3C', TPvector3C)

    TPvector3T = TPvector3C / len(TPvector2)  # Table of probabilities.
    #print ('TPvector3T.type', type(TPvector3T))
    #print ('TPvector3T', TPvector3T)

    # ----------------#
    # 4. p(Xn+s,Xn): #
    # ----------------#

    TPvector4 = []

    for i in range(L4):
        vect = (X[i + s], "i", X[i])
        TPvector4.append(paste(vect, collapse=""))  # "addresses"

    #print ('TPvector4', TPvector4)
    #print ('len(TPvector4)', len(TPvector4))

    ps = pd.Series([tuple(i) for i in TPvector4])
    TPvector4C = ps.value_counts()
    #print ('TPvector4C.type', type(TPvector4C))
    #print ('TPvector4C', TPvector4C)

    TPvector4T = TPvector4C / len(TPvector4)  # Table of probabilities.
    #print ('TPvector4T.type', type(TPvector4T))
    #print ('TPvector4T', TPvector4T)

    # --------------------------#
    # Transfer entropy T(Y->X) #
    # --------------------------#

    # SUMvector=rep(0,length(TPvector1T))
    SUMvector = []

    '''
    print TPvector1T.index[0]
    print (''.join(TPvector1T.index[n])).split('i')
    print type((''.join(TPvector1T.index[n])).split('i')[1])
    print TPvector2T[int(((''.join(TPvector1T.index[n])).split('i'))[1])]
    '''

    for n in range(len(TPvector1T)):
        val = (TPvector1T[n] * TPvector2T[int(((''.join(TPvector1T.index[n])).split('i'))[1])]) / (TPvector3T[tuple((((''.join(TPvector1T.index[n])).split('i'))[1], "i", ((''.join(TPvector1T.index[n])).split('i'))[2]))] * TPvector4T[tuple((((''.join(TPvector1T.index[n])).split('i'))[0], "i", ((''.join(TPvector1T.index[n])).split('i'))[0]))])
        SUMvector.append(TPvector1T[n] * math.log10(val))

    SHEntropy = sum(SUMvector)
    #print SUMvector
    #print ('ans', ans)
    return SHEntropy

def surrogate(ts):
    #np.random.seed(0)
    ts_fourier = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, np.pi, len(ts) / 2 + 1) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.fft.irfft(ts_fourier_new)
    return new_ts

def calculate_plv_threshold(data):
    plv = []

    for i in range(len(data) - 8):
        for j in range(len(data) - 8):
            old_y1 = data[i]
            y1 = surrogate(old_y1)
            old_y2 = data[j]
            y2 = surrogate(old_y2)

            phase1 = np.angle(y1)
            phase2 = np.angle(y2)

            complex_phase_diff = np.exp(np.complex(0, 1) * (phase1 - phase2))
            plv1 = np.abs(np.sum(complex_phase_diff)) / len(phase1)
            plv.append(plv1)

    plv_mean = np.mean(plv)
    plv_std = np.std(plv)
    thres = plv_mean + (2.32 * plv_std)
    return thres

def normalize_data_matrix(data, a = 0, b = 1):
    min_val = np.min(data)
    max_val = np.max(data)

    for i in range(len(data)):
        for j in range(len(data)):
            data[(i,j)] = a + (((data[(i,j)] - min_val)*(b-a))/(max_val-min_val))

    return data

def normalize_data(data, a = 0, b = 1):
    min_val = np.min(data)
    max_val = np.max(data)

    for i in range(len(data)):
        for j in range(len(data)):
            data[i][j] = a + (((data[i][j] - min_val)*(b-a))/(max_val-min_val))

    return data

def sum_weights(data):
    return np.sum(data)

def out_degree(data):
    result = np.zeros(len(data))

    for i in range(len(data)):
        for j in range(len(data)):
            if (data[(i,j)] > 0.0):
                result[i] = result[i] + 1

    return result

def in_degree(data):
    result = np.zeros(len(data))

    for j in range(len(data)):
        for i in range(len(data)):
            if (data[(i,j)] > 0.0):
                result[j] = result[j] + 1

    return result

def weighted_out_degree(data):
    result = np.zeros(len(data))

    for i in range(len(data)):
        for j in range(len(data)):
            if (data[(i,j)] > 0.0):
                result[i] = result[i] + data[(i,j)]

    return result

def weighted_in_degree(data):
    result = np.zeros(len(data))

    for j in range(len(data)):
        for i in range(len(data)):
            if (data[(i,j)] > 0.0):
                result[j] = result[j] + data[(i,j)]

    return result

def shortest_path_length(G, a=None, b=None):
    return nx.shortest_path_length(G, source=a, target=b, weight=None)


def shortest_weighted_path_length(G, a=None, b=None):
    return nx.shortest_path_length(G, source=a, target=b, weight='weight')
    #return nx.dijkstra_path_length(G,source=a,target=b)

def triangles_undirected_graph(G,nodes=None): #for undirected graphs
    #nodes = (0,1)
    return list(nx.triangles(G, nodes).values())

def triangles_directed_graph(data):
    cnt = 0
    for i in range(len(data)):
        for j in range(len(data)):
            for k in range(len(data)):
                lst = []
                if (data[(i,j)] > 0.0 and data[(j,k)] > 0.0 and data[(k,i)] > 0.0):
                    cnt = cnt + 1

    cnt = cnt / 3
    return cnt

def triangles_directed_per_node(data):
    mat = np.zeros(data.shape)

    for i in range(len(data)):
        for j in range(len(data)):
            if (data[(i,j)] > 0.0):
                mat[i][j] = 1

    result = np.zeros(len(data))

    for i in range(len(data)):
        ans = 0
        for j in range(len(data)):
            if (i != j and mat[i][j] > 0):
                for k in range(len(data)):
                    if (j != k and k != i and mat[j][k] > 0):
                        ans += ((mat[i][j] + mat[j][i])*(mat[j][k] + mat[k][j])*(mat[i][k] + mat[k][i]))
        result[i] = ans/2

    return result

def triangles_directed_per_node_weighted(data):
    result = np.zeros(len(data))
    for i in range(len(data)):
        ans = 0
        for j in range(len(data)):
            if (i != j):
                for k in range(len(data)):
                    if (j != k and k != i):
                        ans += np.cbrt(data[(i,j)] * data[(j,k)] * data[(k,i)])
        result[i] = ans/2

    return result

def weighted_characteristic_path_length(G):
    return nx.average_shortest_path_length(G, weight='weight')


def directed_characteristic_path_length(G):
    return nx.average_shortest_path_length(G)

def average_inverse_shortest_path_length(G, weight=None):
    if G.is_directed():
        if not nx.is_weakly_connected(G):
            raise nx.NetworkXError("Graph is not connected.")
    else:
        if not nx.is_connected(G):
            raise nx.NetworkXError("Graph is not connected.")
    avg = 0.0
    if weight is None:
        for node in G:
            path_length = nx.single_source_shortest_path_length(G, node)
            avg += sum(np.reciprocal(path_length.values()))
    else:
        for node in G:
            path_length = nx.single_source_dijkstra_path_length(G, node, weight=weight)
            avg += sum(np.reciprocal(path_length.values()))
    n = len(G)
    return avg / (n * (n - 1))

def global_efficiency_directed(G):
    return average_inverse_shortest_path_length(G, weight=None)

def global_efficiency_weighted(G):
    return average_inverse_shortest_path_length(G, weight='weight')

def clustering_coefficient(data):
    n = len(data)
    ans = 0
    mat = np.zeros(data.shape)

    for i in range(n):
        for j in range(n):
            if (data[(i, j)] > 0.0):
                mat[i][j] = 1

    t = triangles_directed_per_node(data)
    ind = in_degree(data)
    outd = out_degree(data)

    for i in range(n):
        cnt = 0
        for j in range(n):
            if (i != j):
                cnt += (mat[i][j] * mat[j][i])
        ans += (t[i]/(((outd[i] + ind[i])*(outd[i] + ind[i] - 1)) - (2*cnt)))

    ans = ans / n

    return ans


def clustering_coefficient_weighted(G,nodes=None,weight='weight'):
    dict = nx.clustering(G,nodes,weight)
    ans = list(dict.values())

    return np.mean(ans)

def weighted_transitivity(Gu, data):
    deg = Gu.degree()
    t = triangles_directed_per_node_weighted(data)
    t_sum = np.sum(t)
    t_sum = t_sum * 2

    cnt = 0

    for i in deg.keys():
        cnt += (deg[i] * (deg[i] - 1))

    ans = t_sum / cnt
    return ans

def directed_transitivity(data):
    n = len(data)
    ans = 0
    mat = np.zeros(data.shape)

    for i in range(n):
        for j in range(n):
            if (data[(i, j)] > 0.0):
                mat[i][j] = 1

    t = triangles_directed_per_node(data)
    t_sum = np.sum(t)
    ind = in_degree(data)
    outd = out_degree(data)

    for i in range(n):
        cnt = 0
        for j in range(n):
            if (i != j):
                cnt += (mat[i][j] * mat[j][i])
        ans += (((outd[i] + ind[i]) * (outd[i] + ind[i] - 1)) - (2 * cnt))

    res = t_sum / ans

    return res

def efficiency(G, u, v):
    return 1 / nx.shortest_path_length(G, u, v)

def global_efficiency(G):
    n = len(G)
    denom = n * (n - 1)
    return sum(efficiency(G, u, v) for u, v in permutations(G, 2)) / denom

def local_efficiency(G):
    return sum(global_efficiency(nx.ego_graph(G, v)) for v in G) / len(G)

def weighted_local_efficiency(G, Gu, data):
    dict = Gu.degree()
    deg = list(dict.values())
    ans = 0

    for i in range(len(data)):
        cnt = 0
        for j in range(len(data)):
            if (i != j):
                for k in range(len(data)):
                    if (j != k):
                        cnt += np.cbrt(((data[(i,j)] + data[(j,i)]) * (data[(i,k)] + data[(k,i)])) / nx.shortest_path_length(Gu, ch[j], ch[k], weight='weight'))

        a = deg[i] * (deg[i] - 1)
        ans += (cnt / a)

    ans = ans / 2

    return ans

def directed_local_efficiency(G, Gu, data):
    n = len(data)
    ans = 0
    mat = np.zeros(data.shape)
    ind = in_degree(data)
    outd = out_degree(data)

    for i in range(n):
        for j in range(n):
            if (data[(i, j)] > 0.0):
                mat[i][j] = 1


    for i in range(n):
        cnt = 0
        for j in range(len(data)):
            if (i != j):
                for k in range(len(data)):
                    if (j != k):
                        cnt += ((mat[i][j] + mat[j][i]) * (mat[i][k] + mat[k][i]) * (2 / nx.shortest_path_length(Gu, ch[j], ch[k])))

        cnt2 = 0
        for j in range(n):
            if (i != j):
                cnt2 += (mat[i][j] * mat[j][i])
        a = (((outd[i] + ind[i]) * (outd[i] + ind[i] - 1)) - (2 * cnt2))

        ans += (cnt / a)

    ans = (ans / (2 * n))

    return ans




