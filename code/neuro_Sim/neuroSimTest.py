from __future__ import division

from neuro import *

import numpy as np
import scipy as scp
import matplotlib as mp
import matplotlib.pyplot as plt

plt.switch_backend('QT4Agg')

# Global parameters ------------------------------------------------------------
mili  = 0.001         # Scaling factor 10^-3

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
T3    = 0.5
t_S   = 10*mili       # Time scale of the synapse
tau_vR = 12*mili

# Network connectivity structure -----------------------------------------------
#weight matrix := 16x16, layered full connectivity
N  = 16
L  = N/4
cB = np.roll(np.eye(L), 1, 1)            #4x4 box repeating in the matrix
cB[:,0] = 0
cR   = np.concatenate((cB,cB,cB,cB), 1)    #row of boxes
cMat = np.concatenate((cR,cR,cR,cR))       #full matrix
cMat *= 0.6                                #scaled by constant synapse weight


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
  Neurons[i].i = 72*mili if i % L == 0 else I_e

sim2 = netSim(Neurons, cMat, T2, dt, h_t = T1)
Vs_2, Gs_2, raster_2 = sim2.simulate()

for i in range(N):
  Neurons[i].i = I_e

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
n0_vR = vR_mapTrain(Neurons[0].sTrain, dt, T, tau_vR, mu)
n1_vR = vR_mapTrain(Neurons[1].sTrain, dt, T, tau_vR, mu)

dist_01_vR, diff_01_vR = vR_computeDistance_mappedTrains(n0_vR, n1_vR, dt);
print "numerical dist_van_Rossum(nrn_0, nrn_1) = ", dist_01_vR

dist_01_vR_2 = vR_computeDistance(Neurons[0].sTrain, Neurons[1].sTrain, tau_vR)
print "analythic dist_van_Rossum(nrn_0, nrn_1) = ", dist_01_vR_2


# Compute Victor Purpura distance-----------------------------------------------
q = 200
dist_01_VP = VP_computeDistance(Neurons[0].sTrain, Neurons[1].sTrain, q)

print "dist_VictorPurpura(nrn_0, nrn_1) = ", dist_01_VP



# Plot Graphs-------------------------------------------------------------------

#plot voltages of neuron 1 & 2
plt.figure(1)
p1, = plt.plot(time, Vs[0], 'g')
p2, = plt.plot(time, Vs[1], 'r')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$V_m$', fontsize=15)
plt.show()


#plot conductances of neuron 1 & 2
plt.figure(2)
p1, = plt.plot(time, Gs[0], 'g')
p2, = plt.plot(time, Gs[1], 'r')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$G_s$', fontsize=15)
plt.show()


#plot raster

plt.figure(3, dpi=120)
plt.ylim([-1, N])
plt.yticks(np.arange( 0, N, 1))
plt.xticks(np.arange(0, T, 0.1))
p3 = plt.plot(time, np.transpose(raster), 'b.')
plt.grid()
plt.ylabel('$Neuron$ $indices$', fontsize=20)
plt.xlabel('$Spike$ $times$ $[s]$', fontsize=20)
plt.show()


#plot van Rossum functions
f, ax = plt.subplots(3, sharex=True, dpi=120)
ax[0].plot(time, n0_vR, 'g')
#ax[0].set_xlabel('$t$ $[s]$', fontsize=20)
ax[0].set_ylabel('$van$ $Rossum$ \n $(n_0)$', fontsize=20)
ax[1].plot(time, n1_vR, 'b')
#ax[1].set_xlabel('$t$ $[s]$', fontsize=20)
ax[1].set_ylabel('$van$ $Rossum$  \n $(n_1)$', fontsize=20)
ax[2].plot(time, diff_01_vR, 'r')
ax[2].set_xlabel('$t$ $[s]$', fontsize=20)
ax[2].set_ylabel('$van$ $Rossum$ \n $(n_0-n_1)$', fontsize=20)
plt.savefig("vR_savefigtest.pdf", bbox_inches='tight', pad_inches=0.5)
plt.show()

# Test Poisson distribution spike generation
T = 1
counts=[]
for i in range(100):
  t, c = getPoissonTrain(T, dt, 120)
  counts += [c]

maxim = max(counts)
minim = min(counts)

plt.figure(4, dpi=100)
plt.hist(counts, bins=(maxim-minim+1), range=(minim, maxim))
plt.xlabel('$Spiking$ $rate$ $r$', fontsize=20)
plt.ylabel('$Number$ $of$ $occurances$ $\\propto$ $P$ $(r)$', fontsize=20)
plt.show()
