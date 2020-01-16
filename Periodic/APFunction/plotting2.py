import numpy as np
import matplotlib.pyplot as plt
import math

def logbaseten(x):
  return math.log(x,10)

log_v = np.vectorize(logbaseten)

data4 = np.loadtxt('PeriodicApproxAPFunction60*i.dat')

fig,ax = plt.subplots(dpi=150)
plt.xlabel('log R')
plt.ylabel('log Error')

plt.plot(log_v(data4[:,0]), log_v(data4[:,2]), linewidth = 1, marker = "x", markersize = 3)
plt.show()
