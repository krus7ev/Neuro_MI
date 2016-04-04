from __future__ import division
import numpy as np
import math
import scipy as scp
from pylab import *
# ==============================================================================

#Initalise bucket equation parameters    
E_l = -70/ (10**3)      # EL = Vreset = -70 [mV]
t_m  = 10/(10**3)       # Tm = 10 [ms] 
V_th = -40/(10**3)      # Vth = -40 [mV]
V_res = E_l
R_m  = 10*10**6;        # Rm = 10 [M_ohm]
I_e  = 3.1/(10**9)      # Ie = 3.1 [nA]
dt = 0.001 
T = 1 

# Interate and fire model of a single neuron
# ==============================================================================
#   -> Solve the bucket equation numerically using Euler's method
#   -> Reset V to E_l whenever V_th is reached
def intFire(E_l, t_m, V_th, V_res, R_m, I, T, dt):
    V = [V_res] # V(0) = E_l 
    n = int(T/dt+1)
    fire_rate = 0
    for t in range(1,n):
        v = V[-1] + dt*f(V[-1], t_m, E_l, R_m, I)
        
        if v > V_th :
            v = V_res
            fire_rate = fire_rate + 1
            
        V += [v]
    return {'V' : np.array(V), 'fire_rate' : fire_rate}

# Return the value of f(V) = dV/dt at a particular value of V
def f(V, t_m, E_l, R_m, I):
    return (E_l + R_m*I - V)/t_m

    
# ================================== 1 =========================================    
# Simulate a integrate and fire model over 1s 
V = intFire(E_l, t_m, V_th, V_res, R_m, I_e, T, dt)['V']

# np.array to store the sampling points over period T
nt = np.linspace(0, T, T/dt+1)

# Plot the voltage as function of time
figure()
plot(nt,V)
show()

# ================================== 2 =========================================
# Minimum current for neuron with inital parameters to produce a spike:
# 1)Set the potential higher than the threshold
#   E_l + R_m * I_e >= V_th
# 2)Rearange to express I_e
#   I_e >= (V_th - E_l) / R_m
# 3)Substituting in
#   I_e >= 3 * 10^(-9)
I_min = (3-0.1)/(10**9)
# Simulate
V = intFire(E_l, t_m, V_th, V_res, R_m, I_min, T, dt)['V']
figure()
plot (nt,V)
show()

# ================================== 3 =========================================
# Simulate for currents ranging from 2.0 to 5.0 at step 0.1
# Collect number of spikes fired at each iteration
# Plot fire rate as function of current
f_rates = []
for i in range (20,51):
    I_e = i/10**10
    f_rates += [intFire(E_l, t_m, V_th, V_res, R_m, I_e, T, dt)['fire_rate']]
Is = np.linspace(2.0, 5.0, 31)
print len(f_rates)
figure()
plot(Is, f_rates)
show()

