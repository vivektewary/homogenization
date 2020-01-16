import numpy as np
import matplotlib.pyplot as plt
import math

def logbaseten(x):
  return math.log(x,10)

log_v = np.vectorize(logbaseten)

data4 = np.loadtxt('DirichletvsPeriodicAPFunction(100+R*R).dat')
#data5 = np.loadtxt('DirichletvsPeriodicAPFunction(100+2*R*R).dat')
#data6 = np.loadtxt('DirichletvsPeriodicAPFunction(100+3*R*R).dat')

fig,ax = plt.subplots(dpi=150)
plt.xlabel('log R')
plt.ylabel('log E(R)')

plt.plot(log_v(data4[:,0]), log_v(data4[:,3]), linewidth = 1, marker = "o", markersize = 3)
#plt.plot(log_v(data5[:,0]), log_v(data5[:,3]), linewidth = 1, marker = "D", markersize = 3)
#plt.plot(log_v(data6[:,0]), log_v(data6[:,3]), linewidth = 1, marker = "D", markersize = 3)
plt.show()
