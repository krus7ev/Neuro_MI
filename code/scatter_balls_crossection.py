import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
from math import *

plt.switch_backend('QT4Agg')

fig = plt.figure(dpi=200)

N=101
x = np.random.rand(N)
y = np.random.rand(N)

x_med = np.median(x)

idx = [item for sublist in np.nonzero(x == np.median(x)) for item in sublist]
idx = idx[0]

plt.ylim(-0.2,1.2)
plt.xlim(-0.2,1.2)


plot = plt.scatter(x,y,color='black',s=5,edgecolor='none')

plt.axhspan(-0.2, 1.2, x[idx]-0.15, x[idx]+0.15, facecolor='0.8', alpha=0.3, edgecolor='0.7')

plt.axhspan(y[idx]-0.15, y[idx]+0.15, -0.2, 1.2, facecolor='0.8', alpha=0.3, edgecolor='0.7')

plt.axhspan(y[idx]-0.15, y[idx]+0.15, x[idx]-0.15, x[idx]+0.15, alpha=1, edgecolor='0.0', fill = False)

plt.scatter(x[idx],y[idx],facecolor='none', edgecolor='0.0')



plt.annotate( s='$B(s_i,r_i,h_1/N,h_2/N)$', xy=(x[idx]+0.15, y[idx]+0.15), xycoords='data',
                xytext=(90, 60), textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.3") )

plt.annotate( s='$(r_i,s_i)$', xy=(x[idx]+0.01, y[idx]-0.02))


#plt.annotate( s='', xy=(x[idx]-0.15, 1.15), xytext=(x[idx]+0.15, 1.15), arrowprops=dict(arrowstyle="<->") )
plt.annotate( s='$h_1$', xy=(x[idx], 1.1))

#plt.annotate( s='', xy=(1.15, y[idx]-0.15), xytext=(1.15, y[idx]+0.15,), arrowprops=dict(arrowstyle="<->") )
plt.annotate( s='$h_2$', xy=(1.1, y[idx]))




plot.axes.get_xaxis().set_ticks([])
plot.axes.get_yaxis().set_ticks([])
plot.axes.spines['top'].set_color('none')
plot.axes.spines['right'].set_color('none')
plt.xlabel('$S$', fontsize=15)
plt.ylabel('$R$', fontsize=15)


mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()
