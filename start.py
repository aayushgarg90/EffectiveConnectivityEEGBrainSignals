import cPickle
import numpy as np
import statsmodels.tsa.stattools as tsa_stats
import statsmodels.tsa.vector_ar.var_model as vm
import pylab
import graph
import pandas as pd

path2 = "/Users/praneelrathore/PycharmProjects/Project_8_sem/data/"
file2 = "female_02_baseline-slorTransposed.txt"
find = path2 + file2
data = pd.DataFrame.as_matrix(pd.read_csv(find,sep = "  ", header = None))
print data.shape
#data = pd.DataFrame.as_matrix(pd.read_csv("/Users/praneelrathore/PycharmProjects/Project_8_sem/female_02_baseline-slorTransposed.txt", sep="  ", header = None))
#data = data.T

testarray = data[:,[0,1]]
'''
model = vm.VAR(data,missing='drop')
results = model.fit(2, ic = 'aic')
print results.summary()
'''
#ge = tsa_stats.grangercausalitytests(testarray[:,1::-1],1,verbose=False)
#print ge

ls=[]

for i in range(1,384):
    #print i
    testarray = data[:,[i-1,i]]
    gr1 = tsa_stats.grangercausalitytests(testarray[:, :], 1, verbose=False)
    gr = tsa_stats.grangercausalitytests(testarray[:, 1::-1], 1, verbose=False)

    local = []
    for key in gr:
        if gr[key][0]['ssr_ftest'][1] > 0.5:
            local.append(i-1)
            local.append(i)
            local.append(gr[key][0]['ssr_ftest'][1])
        if local:
            ls.append(local)
    local2 = []
    for key in gr1:
        if gr1[key][0]['ssr_ftest'][1] > 0.5:
            local2.append(i)
            local2.append(i-1)
            local2.append(gr1[key][0]['ssr_ftest'][1])
        if local2:
            ls.append(local2)
            # print ls
graph.draw(ls)



'''
path = "/Users/praneelrathore/PycharmProjects/Project_8_sem/data/data_preprocessed_python/"
file = "s01.dat"
op = path +file
p = cPickle.load(open(op,'rb'))
testdata = p['data']

for j in range(0,32):
    trial = testdata[j,0:32]                #trial.shape = [32,8064]
    trial_transpose = np.transpose(trial)
    #testarray = trial_transpose[:,[24,25]]
    #gr = tsa_stats.grangercausalitytests(testarray[:, 1::-1], 1, verbose=False)
    #for key in gr:
    #    print gr[key][0]['ssr_ftest'][1]

    #model = vm.VAR(testarray)
    #results = model.fit(2, ic = 'aic')
    #print results.summary()

    #k= model.select_order(10)
    #print k

    ls = []
    for i in range(0,31):
        testarray = trial_transpose[:,[i,i+1]]
        gr1 = tsa_stats.grangercausalitytests(testarray[:,:], 1, verbose=False)
        gr = tsa_stats.grangercausalitytests(testarray[:,1::-1], 1, verbose=False)
        local=[]
        for key in gr:
            if gr[key][0]['ssr_ftest'][1] > 0.5:
                local.append(i)
                local.append(i+1)
                local.append(gr[key][0]['ssr_ftest'][1])
            if local:
                ls.append(local)
        local2=[]
        for key in gr1:
            if gr1[key][0]['ssr_ftest'][1] > 0.5:
                local2.append(i+1)
                local2.append(i)
                local2.append(gr1[key][0]['ssr_ftest'][1])
            if local2:
                ls.append(local2)
    #print ls
    graph.draw(ls)

'''

