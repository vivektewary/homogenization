import numpy as np
import matplotlib.pyplot as plt
import math

def logbaseten(x):
  return math.log(x,10)

log_v = np.vectorize(logbaseten)

data1 = np.loadtxt('PeriodicApproxFunction2P2(100+80*R).dat')

fig,ax = plt.subplots(dpi=150)
plt.xlabel('log R')
plt.ylabel('log Error')

plt.plot(log_v(data1[:,0]), log_v(data1[:,2]), linewidth = 1, marker = "o", markersize = 3)
plt.show()
