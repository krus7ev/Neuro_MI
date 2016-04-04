import numpy as np
import scipy as scp
from math  import exp
from scipy.integrate import odeint
from pylab import *

def df(f,t):  #return derivatives of the array f
    a = 3.0
    return (f**2 - a*f + exp(-t))
    
t1 = linspace(0.0, 3.0, 301)  #integrate over[0,3] with DT=0.01
t2 = linspace(0.0, 3.0, 31)   #integrate over[0,3] with DT=0.1
t3 = linspace(0.0, 3.0, 7)    #integrate over[0,3] with DT=0.5
t4 = linspace(0.0, 3.0, 4)    #integrate over[0,3] with DT=1.0

fInit = 0;                    # initial value
f1 = odeint(df, fInit, t1)
f2 = odeint(df, fInit, t2)
f3 = odeint(df, fInit, t3)
f4 = odeint(df, fInit, t4)

figure()
plot(t1,f1)
plot(t2,f2)
plot(t3,f3)
plot(t4,f4)

xlabel('t')
ylabel('f')
show()
