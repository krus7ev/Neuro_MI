from __future__ import division
import numpy as np
import scipy as scp
import matplotlib as mp
import matplotlib.pyplot as plt

from pylab import *
from neuro import *
    
# Global parameters ------------------------------------------------------------
mili  = 0.001          # Scaling factor 10^-3

E_l   = -70*mili      # Standard reverse potential 
V_th  = -54*mili      # Vth = -40 [mV]
V_res = -80*mili      # Reset membrane potential
R_m   = 1             # Rm = 1[M_ohm] 
I_e   = 18*mili       # Ie = 3.1[nA]
t_M   = 30*mili       # tau_m = 10[ms] = C_m*Rm time constant of the membrane
t_Ref = 5*mili        # Refractory period = 5[ms]

dt    = 1*mili        # Time scale [ms]
T1    = 0.3           # simulation period [s]
T2    = 0.2
T3    = 1
t_S   = 10*mili       # Time scale of the synapse


# Network connectivity structure -----------------------------------------------
#weight matrix := 16x16, layered full connectivity
N = 16
L = N/4
cB   = np.roll(np.eye(L), 1, 1)            #4x4 box repeating in the matrix
cB[:,0] = 0
cR   = np.concatenate((cB,cB,cB,cB), 1)    #row of boxes
cMat = np.concatenate((cR,cR,cR,cR))       #full matrix
cMat *= 0.3                                #scaled by constant synapse weight


# Initialise neuron vector------------------------------------------------------
#generate random initial membrane voltages and last-spike times
Vs = V_res + np.random.rand(N)*(V_th - V_res)
sTs = -t_M + np.random.rand(N)*t_M

Neurons = []    
for i in range(N) :
  Neurons.append(neuron(i, Vs[i], sTs[i], E_l, V_th, V_res, R_m, I_e, t_M, t_Ref))


# Network simulation------------------------------------------------------------
#1st part - T1
sim1 = netSim(Neurons, cMat, T1, dt)
Vs_1, Gs_1, raster_1 = sim1.simulate()

#2nd part - T2
for i in range(N):
  Neurons[i].i_ext = 72*mili if i % L == 0 else I_e
 
sim2 = netSim(Neurons, cMat, T2, dt, h_t = T1)
Vs_2, Gs_2, raster_2 = sim2.simulate()

for i in range(N):
  Neurons[i].i_ext = I_e

#3nd part - T3
sim3 = netSim(Neurons, cMat, T3, dt, h_t = T1+T2)
Vs_3, Gs_3, raster_3 = sim3.simulate()


# Assemble results--------------------------------------------------------------
T = T1 + T2 + T3
time = np.arange(0, T, dt)
Vs = np.concatenate((Vs_1, Vs_2, Vs_3),1)
Gs = np.concatenate((Gs_1, Gs_2, Gs_3),1)
raster = np.concatenate((raster_1, raster_2, raster_3), 1)

#print spike trains
for i in range(2):
  print [round(j, 2) for j in Neurons[i].sTrain]


# Compute van Rossum distance---------------------------------------------------
# convert spike trains to functions
mu = 0
n0_vR = vR_mapTrain(Neurons[0].sTrain, dt, T, t_S, mu)
n1_vR = vR_mapTrain(Neurons[1].sTrain, dt, T, t_S, mu)

dist_01_vR, diff_01_vR = vR_computeDistance(n0_vR, n1_vR, dt);
print "dist_van_Rossum(nrn_0, nrn_1) = ", dist_01_vR


# Compute Victor Purpura distance-----------------------------------------------
q = 200
dist_01_VP = VP_computeDistance(Neurons[0].sTrain, Neurons[1].sTrain, q)

print "dist_VictorPurpura(nrn_0, nrn_1) = ", dist_01_VP



# Plot Graphs-------------------------------------------------------------------

#plot voltages of neuron 1 & 2
figure(1)
p1, = plot(time, Vs[0], 'g')  
p2, = plot(time, Vs[1], 'r')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$V_m$', fontsize=15)
show()

'''
#plot condictances of neuron 1 & 2
figure(2)
p1, = plot(time, Gs[0], 'g')  
p2, = plot(time, Gs[1], 'r')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$G_s$', fontsize=15)
show()
'''

#plot raster
figure(3)
ylim([-1, N])
yticks(np.arange( 0, N, 1))
xticks(np.arange(0, T, 0.05))
p3 = plot(time, np.transpose(raster), 'b.')
plt.grid()
plt.ylabel('$Neuron$ $indices$', fontsize=20)
plt.xlabel('$Spike$ $times$ $[s]$', fontsize=20)
show()


#plot van Rossum functions
f, ax = plt.subplots(3, sharex=True)
ax[0].plot(time, n0_vR, 'g')
#ax[0].set_xlabel('$t$ $[s]$', fontsize=20)
ax[0].set_ylabel('$van$ $Rossum$ \n $(n_0)$', fontsize=20)
ax[1].plot(time, n1_vR, 'b')
#ax[1].set_xlabel('$t$ $[s]$', fontsize=20)
ax[1].set_ylabel('$van$ $Rossum$  \n $(n_1)$', fontsize=20)
ax[2].plot(time, diff_01_vR, 'r')
ax[2].set_xlabel('$t$ $[s]$', fontsize=20)
ax[2].set_ylabel('$van$ $Rossum$ \n $(n_0-n_1)$', fontsize=20)
plt.show()


''' TEST FOR POISSON SPIKES
T = 1
counts=[]
for i in range(100):
  t, c = getPoissonTrain(T, dt, 100)
  counts += [c]
  
maxim = max(counts)
minim = min(counts)

plt.figure(4)
plt.hist(counts, bins=(maxim-minim+1), range=(minim, maxim))
plt.show()
'''
