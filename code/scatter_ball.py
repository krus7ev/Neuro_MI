import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from math import *

plt.switch_backend('QT4Agg') #default on my system

fig = plt.figure(dpi=250)
plt.ylim(-0.2,1.2)
plt.xlim(-0.2,2.2)


N=51
x = np.random.rand(N)*2
y = np.random.rand(N)

x_med = np.median(x)

idx = [item for sublist in np.nonzero(x == np.median(x)) for item in sublist]
idx = idx[0]

plot = plt.scatter(x,y,color='black',s=5,edgecolor='none')

circle1 = plt.Circle( (x[idx], y[idx]), .3, facecolor='1.0', alpha=0.3, edgecolor='0.0')
fig = plt.gcf()
fig.gca().add_artist(circle1)

plt.scatter(x[idx],y[idx],facecolor='none', edgecolor='0.0')


ptr_x = x[idx] + 0.3*cos(1)
ptr_y = y[idx] + 0.3*sin(1)


rad_x = x[idx] + 0.29*cos(2)
rad_y = y[idx] + 0.29*sin(2)

mid_x = rad_x + (x[idx] - rad_x)/2
mid_y = y[idx] + (rad_y - y[idx])/2


plt.annotate( s='$\epsilon$', xy=(mid_x, mid_y))
plt.annotate( s='$x_i$', xy=(x[idx]+0.02, y[idx]-0.03))

plt.annotate('$\mathcal{X}$', xy=(-0.1,-0.1))

plt.annotate( s='$B_{\epsilon}(x_i)$', xy=(ptr_x, ptr_y), xycoords='data',
                xytext=(60, 40), textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.3") )

plt.annotate( s='', xy=(x[idx]-0.005, y[idx]+0.015), xytext=(rad_x, rad_y), arrowprops=dict(arrowstyle="<->") )


plot.axes.get_xaxis().set_ticks([])
plot.axes.get_yaxis().set_ticks([])
plot.axes.spines['top'].set_color('none')
plot.axes.spines['right'].set_color('none')
plot.axes.spines['bottom'].set_color('none')
plot.axes.spines['left'].set_color('none')


mng = plt.get_current_fig_manager()
mng.window.showMaximized()

plt.show()
