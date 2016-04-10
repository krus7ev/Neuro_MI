from __future__ import division
import matplotlib.pyplot as plt
import math
import random
from numpy import *
import pylab
# Parameters for both the Models
tau_m = 20           # (ms)
El    = -70          # (mV)
Vr    = -80          # (mV)
Vt    = -54          # (mV)
RmIe  = 18           #  (mV)

#Parameters for both the Synapses
Rmgs = 0.15
Ps = 0.54
tau_s = 10 
Vm_1 = []
Vm_2 = []

# Case 1. Assuming synapses are excitatory
Es = 0                        # (mV)
dt = 1.0                         #(ms)
t_f1 = 0
t_f2 = 0
time_stamp1 = 0
time_stamp2 = 0

V1 = random.uniform(-80,-54)
V2 = random.uniform(-80,-54)
time = []
Va = []
Vb = []

# Calculate the synaptic voltage for both the neurons
T = 0
while T < 1000:
    time.append(T)
    
    
    dV1 = (El - V1 - (((Ps*Rmgs*T) *exp(-(T-t_f1)/tau_s)* (V1-Es)) +(RmIe))/tau_m)*dt
    Vm_1 =  V1 + dV1
    
    
    if Vm_1 >= Vt:
        #time_stamp1 = T - t_f1
        Vm_1 = Vr
        t_f1 = T #time of the spike
        
    
    dV2 = (El -V2 - (((Ps*Rmgs*T)*(exp(-(T-t_f2)/tau_s)) * (V2-Es)) +(RmIe))/tau_m)*dt
    Vm_2 = V2 + dV2
    #time_stamp2 = T - t_f2
    
    
    if Vm_2 >= Vt:
        #time_stamp2 = T - t_f2
        Vm_2 = Vr
        t_f2 = T  
        
    
    T = T + dt
    Va.append(Vm_1) 
    Vb.append(Vm_2)
    

plot1 = plt.plot(time,Va,'b',label='Neuron1')
plot2 = plt.plot(time,Vb,'r',label='Neuron2')

    
plt.xlabel('Time(mS)')
plt.ylabel('Voltage (mV)')
plt.title('Excitatory Synapse')

pylab.legend(loc = 'upper right')
plt.show()







