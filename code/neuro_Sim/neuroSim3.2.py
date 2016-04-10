from __future__ import division

from neuro import *

from copy import deepcopy

import numpy as np
import scipy as scp
import matplotlib as mp
import matplotlib.pyplot as plt

plt.switch_backend('QT4Agg')

# EXPERIMENTAL NETWORK CONNECTIVITY
#
# Integrate-and-fre neurons {0,1}
#
#       0     1     
#       ^     ^
#       |\1-g/|
#       | \ / |
#     g |  x  | g        Synapse strength g in [0,1]
#       | / \ |
#       |/   \|
#       2     3
#        
# Poisson neurons {2,3}
   
# Global parameters ------------------------------------------------------------
mili  = 0.001          # Scaling factor 10^-3

E_l   = -70*mili       # Standard reverse potential 
V_th  = -54*mili       # Vth = -40 [mV]
V_res = -80*mili       # Reset membrane potential
R_m   = 1              # Rm = 1[M_ohm] 
I_e   = 18*mili        # Ie = 3.1[nA]
t_M   = 30*mili        # tau_m = 10[ms] = C_m*Rm time constant of the membrane
t_Ref = 5*mili         # Refractory period = 5[ms]

dt    = 1*mili         # Time scale [ms]
T     = 1              # total simulation period [s]

t_S   = 10*mili        # Time scale of the synapse

IF = 2                 # Number of Integrate and Fire neurons
P  = 2                 # Number of Poisson neurons
N  = IF + P            # Total number of neurons

minRate = 10           # maximum average spike-rate for poisson neurons
maxRate = 20           # maximum average spike-rate for poisson neurons
rateRange = maxRate - minRate

trials  = 48           # number of simulation rounds in each experiment
rounds  = 23

d_luft = 12
d_bin  = 0.05          # 10**(-len(str(d_luft)))

h1 = 12                # size of open balls in spike-trrain metric space
h2 = 12

tau_vR = 12*mili
vR_metric = metric('vR', tau=tau_vR)
VP_metric = metric('VP', q=166)

metric = vR_metric

MI_2_0   = []

g = 0.9               # synapse 2-0 and 3-1 strength == 1 minus 2-1 or 3-0

# Experiment Simulation---------------------------------------------------------
# set up network topology
samples = []
cMat = np.array([[  0,   0,  0,  0],
                 [  0,   0,  0,  0],
                 [  g, 1-g,  0,  0],
                 [1-g,   g,  0,  0]])

for r in range(rounds):
  sample = experiment()
# simulate network a number of trials to generate sample data
  for t in range(trials):
#   initialise network parameters
    Vs      = V_res + np.random.rand(IF)*(V_th - V_res)
    sTs     = -t_M + np.random.rand(N)*t_M
    sRates  = minRate + np.random.rand(P)*(rateRange)      
    neurons = []
    for i in range(IF):
      neurons.append(neuron(i, Vs[i], sTs[i], E_l, V_th, V_res, R_m, I_e, t_M, t_Ref))
    for i in range(IF, N):
      pN = pNeuron(i, sRates[i-IF], T, dt, sTs[i])
      pN.delay(0.05)
      neurons.append(pN)

#   simulate network and save data
    sample.simulation.append(netSim(neurons, cMat, T, dt))
    Vs, Gs, raster = sample.simulation[-1].simulate()
    sample.population.append(neurons)
    
  samples.append(sample)

#slide trhough the predefined range of delays
for d in range(-d_luft, d_luft+1) :
  d *= d_bin
# estimate information at each round and each delay
  MI_d = []  
  for s in range(rounds):
  # group together spike trains from each IF neuron over trials
    neuro_var = []
    for n in range(IF):
      s_trains = []
      for t in range(trials):
        s_trains.append(samples[s].population[t][n].sTrain)
      neuro_var.append(s_trains)

  # delay and group together spike trains from each Poisson neuron over trials
    for n in range(IF, N):
      s_trains = []
      for t in range(trials):
        pN = deepcopy(samples[s].population[t][n])
        pN.delay(d)
        s_trains.append(pN.sTrain)
      neuro_var.append(s_trains)

#   compute mutual information between neurons 2 & 0 at current round
    MI_d.append(computeMI(neuro_var[2], neuro_var[0], metric, h1, h2))  
# add list of mutual informations computted at current delay to array  
  MI_2_0.append(MI_d)

#calculate mean MI from samples at each delay point
MI_2_0_avg = []
MI_2_0_std = []
for d in range(len(MI_2_0)):
  MI_d_avg = 0
  for m in range(rounds): 
    MI_d_avg += MI_2_0[d][m]
  MI_d_avg /= rounds
  MI_d_std = 0
  for m in range(rounds):
    MI_d_std += (MI_2_0[d][m] - MI_d_avg)**2
  MI_d_std = np.sqrt(MI_d_std/rounds)
   
  MI_2_0_avg.append(MI_d_avg)
  MI_2_0_std.append(MI_d_std)
  
# Produce Graphs----------------------------------------------------------------

#plot MI
d_lays = [d*d_bin for d in range(-d_luft,d_luft+1)]

plt.figure(2)
#plt.scatter(d_lays, MI_2_0_avg)
plt.errorbar(d_lays, MI_2_0_avg, yerr=MI_2_0_std, fmt='o')
plt.ylabel('$Mutual$ $information$ $I(n_2;n_0)$', fontsize=20)
plt.xlabel('$Poisson$ $spikes$ $delay$ $[s]$', fontsize=20)
plt.show()


#plot raster
time = np.arange(0, T, dt)

plt.figure(1)
plt.ylim([-1, N])
plt.yticks(np.arange( 0, N, 1))
plt.xticks(np.arange(0, T, 0.05))
p3 = plt.plot(time, np.transpose(raster), 'b.')
plt.grid()
plt.ylabel('$Neuron$ $indices$', fontsize=20)
plt.xlabel('$Spike$ $times$ $[s]$', fontsize=20)
plt.show()

