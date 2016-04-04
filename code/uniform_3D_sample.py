import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as a3d


mean  = [0,0,0]
cov   = [[1,0,0],[0,1,0],[0,0,1]]
x,y,z = random.multivariate_normal(mean, cov, 100).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100

for i in range(0,99):
    ax.scatter(x[i],y[i],z[i])

plt.show()



