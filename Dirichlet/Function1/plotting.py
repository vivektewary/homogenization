import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

def logbaseten(x):
  return math.log(x,10)

log_v = np.vectorize(logbaseten)

data = np.loadtxt('DirichletApproxFunction2.dat')

fig,ax = plt.subplots(dpi=150)
plt.xlabel('log R')
plt.ylabel('log $|A^{R,D,*}-A^*|$')
#ax.set_aspect(1)
ax.plot(log_v(data[:,0]), log_v(data[:,3]), color='black', linewidth=1, marker='o', markerfacecolor='r', markeredgecolor = 'r', markersize = 3)

locs = np.append( np.arange(0.1,1,0.1),np.arange(-1,2,0.2))
ax.xaxis.set_minor_locator(ticker.FixedLocator(locs))
#ax.xaxis.set_major_locator(ticker.NullLocator())

#ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())

plt.show()
