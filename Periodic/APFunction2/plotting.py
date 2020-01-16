import numpy as np
import matplotlib.pyplot as plt
import math

def logbaseten(x):
  return math.log(x,10)

log_v = np.vectorize(logbaseten)

data4 = np.loadtxt('PeriodicApproxAPFunction(100+R*R).dat')
data5 = np.loadtxt('PeriodicApproxAPFunction(n+sqrt 2).dat')
data6 = np.loadtxt('PeriodicApproxAPFunction(1+n*sqrt 2).dat')
data7 = np.loadtxt('PeriodicApproxAPFunction(1+n*sqrt 3).dat')

fig,ax = plt.subplots(dpi=150)
plt.xlabel('log R')
plt.ylabel('log Error')

plt.plot(log_v(data4[:,0]), log_v(data4[:,2]), linewidth = 1, marker = "x", markersize = 3)
plt.plot(log_v(data5[:,0]), log_v(data5[:,2]), linewidth = 1, marker = "D", markersize = 3)
plt.plot(log_v(data6[:,0]), log_v(data6[:,2]), linewidth = 1, marker = "*", markersize = 3)
plt.plot(log_v(data7[:,0]), log_v(data7[:,2]), linewidth = 1, marker = "o", markersize = 3)
plt.show()
