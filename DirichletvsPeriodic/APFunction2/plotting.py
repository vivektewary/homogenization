import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt('100+5*i*i.dat')
data2 = np.loadtxt('100+10*i*i.dat')

plt.plot(data1[:,0], data1[:,4])
plt.plot(data2[:,0], data2[:,4])
plt.show()
