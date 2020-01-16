import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

data = np.loadtxt('a22 5*i*i.dat')

plt.plot(data[:,0], data[:,4])
plt.show()
