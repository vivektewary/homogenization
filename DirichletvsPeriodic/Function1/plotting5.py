import numpy as np
import matplotlib.pyplot as plt
import math

def logbaseten(x):
  return math.log(x,10)

log_v = np.vectorize(logbaseten)

data4 = np.loadtxt('DirichletvsPeriodicFunction1(100+R*R).dat')

fig,ax = plt.subplots(dpi=150)
plt.xlabel('log R')
plt.ylabel('log $|A^{R,D,*}-A^{R,*}|$')

plt.plot(log_v(data4[:,0]), log_v(data4[:,3]), 'k' , linewidth = 1, marker = "s", markersize = 3)
plt.show()
