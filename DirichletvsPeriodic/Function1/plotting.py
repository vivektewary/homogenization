import numpy as np
import matplotlib.pyplot as plt

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
