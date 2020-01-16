import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

def logbaseten(x):
  return math.log(x,10)

log_v = np.vectorize(logbaseten)

data = np.loadtxt('DirichletApproxFunction1-secondrun.dat')

fig1,ax = plt.subplots(dpi=300)
ax.plot(log_v(data[:,0]), log_v(data[:,3]))
#ax.set_xscale('log')
#ax.set_yscale('log')

locs = np.append( np.arange(0.1,1,0.1),np.arange(-1,2,0.25))
ax.xaxis.set_minor_locator(ticker.FixedLocator(locs))
#ax.xaxis.set_major_locator(ticker.NullLocator())

#ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())

plt.show()
