import numpy as np
import pylab
import matplotlib.pyplot as plt
import math

def logbaseten(x):
  return math.log(x,10)

log_v = np.vectorize(logbaseten)

data4 = np.loadtxt('PeriodicApproxFunction2(100+R*R).dat')
data5 = np.loadtxt('PeriodicApproxFunction2(100+2*R*R).dat')
data6 = np.loadtxt('PeriodicApproxFunction2(100+3*R*R).dat')

fig,ax = plt.subplots(dpi=150)
plt.xlabel('log R')
plt.ylabel('log $|A^{R,*}-A^*|$')

plt.plot(log_v(data4[:,0]), log_v(data4[:,2]), linewidth = 1, marker = "x", markersize = 3, label='$n=100+R^2$')
plt.plot(log_v(data5[:,0]), log_v(data5[:,2]), linewidth = 1, marker = "D", markersize = 3, label='$n=100+2R^2$')
plt.plot(log_v(data6[:,0]), log_v(data6[:,2]), linewidth = 1, marker = "*", markersize = 3, label='$n=100+3R^2$')
pylab.legend(loc='lower left')
plt.show()
