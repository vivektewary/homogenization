import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('DirichletvsPeriodicAPFunction.dat')

plt.plot(data[:,0], data[:,3])
plt.show()
