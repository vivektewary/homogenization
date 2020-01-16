import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt('DirichletvsPeriodicFunction2(40*i*i).dat')
data2 = np.loadtxt('DirichletvsPeriodicFunction2(30*i*i).dat')
data3 = np.loadtxt('DirichletvsPeriodicFunction2(20*i*i).dat')
data4 = np.loadtxt('DirichletvsPeriodicFunction2(10*i*i).dat')
data5 = np.loadtxt('DirichletvsPeriodicFunction2(5*i*i).dat')

plt.plot(data1[:,0], data1[:,3])
plt.plot(data2[:,0], data2[:,3])
plt.plot(data3[:,0], data3[:,3])
plt.plot(data4[:,0], data4[:,3])
plt.plot(data5[:,0], data5[:,3])
plt.show()
