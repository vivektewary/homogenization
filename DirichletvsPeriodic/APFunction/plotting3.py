import numpy as np
import matplotlib.pyplot as plt
import math

def logbaseten(x):
  return math.log(x,10)

log_v = np.vectorize(logbaseten)

data4 = np.loadtxt('100+10*i*i.dat')
data5 = np.loadtxt('100+5*i*i.dat')

fig,ax = plt.subplots(dpi=150)
plt.xlabel('log R')
plt.ylabel('log Mean of $H^1$ Difference')

plt.plot(log_v(data4[:,0]), log_v(data4[:,5]), linewidth = 1, marker = "x", markersize = 3)
plt.plot(log_v(data5[:,0]), log_v(data5[:,5]), linewidth = 1, marker = "D", markersize = 3)
plt.show()
