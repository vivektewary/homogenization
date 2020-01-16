import numpy as np
import matplotlib.pyplot as plt
import math

def logbaseten(x):
  return math.log(x,10)

log_v = np.vectorize(logbaseten)

data1 = np.loadtxt('DirichletvsPeriodicFunction1.dat')

plt.plot(log_v(data1[:,0]), log_v(data1[:,4]))
plt.show()
