import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import signal
import cPickle
import networkx as nx
import pandas

def surrogate(ts):
    np.random.seed(0)
    ts_fourier = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, np.pi, len(ts) / 2 + 1) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.fft.irfft(ts_fourier_new)
    return new_ts


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

ts = data[0]

ts_fourier  = np.fft.rfft(ts)
random_phases = np.exp(np.random.uniform(0,np.pi,len(ts)/2+1)*1.0j)
ts_fourier_new = ts_fourier*random_phases
new_ts = np.fft.irfft(ts_fourier_new)


plt.plot(ts, label="original data")
plt.plot(new_ts, label="surrogate data")
#plt.ylim([-0.1,0.1])
plt.title('Original v/s Surrogate time series')
plt.legend()

plv = []

for i in range(len(data)-8):
    for j in range(len(data)-8):

        old_y1 = data[i]
        y1 = surrogate(old_y1)
        old_y2 = data[j]
        y2 = surrogate(old_y2)

        phase1 = np.angle(y1)
        phase2 = np.angle(y2)

        complex_phase_diff = np.exp(np.complex(0, 1)*(phase1 - phase2))
        plv1 = np.abs(np.sum(complex_phase_diff))/len(phase1)
        plv.append(plv1)

plv_mean = np.mean(plv)
plv_std = np.std(plv)
thres = plv_mean + (2.32 * plv_std)
print ('plv_mean', plv_mean)
print ('plv_std', plv_std)
print ('thres', thres)  #0.24220825276462854


plt.show()



