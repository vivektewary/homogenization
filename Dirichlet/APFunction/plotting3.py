import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

def logbaseten(x):
  return math.log10(x)

log_v = np.vectorize(logbaseten)

data = np.loadtxt('a22 5*i*i.dat')

fig,ax = plt.subplots(dpi=150)
plt.xlabel('log R')
plt.ylabel('log Error')
#ax.set_aspect(1)
#ax.plot(log_v(data[:,0]), log_v(data[:,2]), color='black', linewidth=1, marker='o', markerfacecolor='r', markeredgecolor = 'r', markersize = 3)
ax.plot(log_v(data[:,0]), log_v(data[:,4]), linestyle = 'None', marker='o', markerfacecolor='r', markeredgecolor = 'r', markersize = 3)

coefficients = np.polyfit(log_v(data[:,0]), log_v(data[:,4]), 1)
polynomial = np.poly1d(coefficients)
log10_y_fit = polynomial(log_v(data[:,0]))

plt.plot(log_v(data[:,0]), log10_y_fit, color='black', linewidth=1)

locs = np.append( np.arange(0.1,1,0.1),np.arange(-1,2,0.2))
ax.xaxis.set_minor_locator(ticker.FixedLocator(locs))
#ax.xaxis.set_major_locator(ticker.NullLocator())

#ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())

plt.show()
