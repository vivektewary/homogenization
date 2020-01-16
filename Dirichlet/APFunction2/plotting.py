import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

data = np.loadtxt('DirichletApproxAPFunction.dat')

plt.plot(data[:,0], data[:,2])
plt.show()
