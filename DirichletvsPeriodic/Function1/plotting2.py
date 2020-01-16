import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

data1 = np.loadtxt('DirichletvsPeriodicFunction1(40*i*i).csv')
data2 = np.loadtxt('DirichletvsPeriodicFunction1(30*i*i).csv')
data3 = np.loadtxt('DirichletvsPeriodicFunction1(20*i*i).csv')
data4 = np.loadtxt('DirichletvsPeriodicFunction1(10*i*i).csv')
data5 = np.loadtxt('DirichletvsPeriodicFunction1(5*i*i).csv')

plt.plot(data1[:,0], data1[:,5])
plt.plot(data2[:,0], data2[:,5])
plt.plot(data3[:,0], data3[:,5])
plt.plot(data4[:,0], data4[:,5])
plt.plot(data5[:,0], data5[:,5])
plt.show()

#def logbaseten(x):
#  return math.log(x,10)

#log_v = np.vectorize(logbaseten)

#data = np.loadtxt('DirichletApproxFunction1-secondrun.dat')

#fig1,ax = plt.subplots(dpi=300)
#ax.plot(log_v(data[:,0]), log_v(data[:,3]))
#ax.set_xscale('log')
#ax.set_yscale('log')

#locs = np.append( np.arange(0.1,1,0.1),np.arange(-1,2,0.25))
#ax.xaxis.set_minor_locator(ticker.FixedLocator(locs))
#ax.xaxis.set_major_locator(ticker.NullLocator())

#ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())

#plt.show()
