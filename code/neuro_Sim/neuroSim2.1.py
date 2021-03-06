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

trials  = 300          # number of simulation rounds in the experiment

delay  = 0.125
d_luft = 60
d_bin  = 10**(-len(str(d_luft)))

h1 = 40                # size of open balls in spike-trrain metric space
h2 = 40

tau_vR = 12*mili
vR_metric = metric('vR', tau=tau_vR)
VP_metric = metric('VP', q=166)

metric = vR_metric

MI_2_0 = []
MI_2_0_Sum = 0

MI_2_1 = []
MI_2_1_Sum = 0

g = 0.9               # synapse 2-0 and 3-1 strength == 1 minus 2-1 or 3-0

# Experiment Simulation---------------------------------------------------------
#set up network topology
sample = experiment()
cMat = np.array([[  0,   0,  0,  0],
                 [  0,   0,  0,  0],
                 [  g, 1-g,  0,  0],
                 [1-g,   g,  0,  0]])

#simulate network a number of trials to generate sample data
for r in range(trials) :
# initialise network parameters
  Vs = V_res + np.random.rand(IF)*(V_th - V_res)
  sTs = -t_M + np.random.rand(N)*t_M
  sRates = minRate + np.random.rand(P)*(rateRange)
  
  neurons = []
  for i in range(IF) :
    neurons.append(neuron(i, Vs[i], sTs[i], E_l, V_th, V_res, R_m, I_e, t_M, t_Ref))
  for i in range(IF, N) :
#   add a predefined delay to poisson neurons spike trains
    pN = pNeuron(i, sRates[i-IF], T, dt, sTs[i])
    pN.delay(delay)
    neurons.append(pN)

# simulate network and save data
  sample.simulation += [netSim(neurons, cMat, T, dt)]
  Vs, Gs, raster = sample.simulation[-1].simulate()

  sample.population += [neurons]
#  sample.VsResults  += [Vs]
#  sample.GsResults  += [Gs]
#  sample.rasters    += [raster]

#group together spike trains elicited by each IF neuron over trials
neuro_var = []
for n in range(IF):
  s_trains = []
  for r in range(trials):
    s_trains += [sample.population[r][n].sTrain]
  neuro_var += [s_trains]
#initialise Poisson-neuron spike trains
for n in range(IF, N):
  neuro_var += [[]]

#slide trhough a predefined range of delays
for d in range(-d_luft, d_luft) :
  d *= d_bin
# add delay to already elicited Poisson trains (with a specific delay)
# and assemble them together by Poisson neuron 
  for n in range(IF, N):
    s_trains = []
    for r in range(trials):
      pN = deepcopy(sample.population[r][n])
      pN.delay(d)
      s_trains += [pN.sTrain]
    neuro_var[n] = s_trains

# compute mutual information between neurons 2 & 0
  MI_2_0 += [computeMI(neuro_var[2], neuro_var[0], metric, h1, h2)]
  MI_2_0_Sum += MI_2_0[-1]
  
#  MI_2_1 += [computeMI(neuro_var[2], neuro_var[1], metric, h1, h2)]
#  MI_2_1_Sum += MI_2_1[-1]

MI_2_0_Avg = MI_2_0_Sum / (2*d_luft)
#MI_2_1_Avg = MI_2_1_Sum / (alpha_res-1)

print 'Average MI(n2;n0) = ' + str(MI_2_0_Avg)
#print 'Average MI(n2;n1) = ' + str(MI_2_1_Avg)


# Produce Graphs----------------------------------------------------------------

#plot raster
time = np.arange(0, T, dt)

plt.figure(1,dpi=120)
plt.ylim([-1, N])
plt.yticks(np.arange( 0, N, 1))
plt.xticks(np.arange(0, T, 0.05))
p3 = plt.plot(time, np.transpose(raster), 'b.')
plt.grid()
plt.ylabel('$Neuron$ $indices$', fontsize=20)
plt.xlabel('$Spike$ $times$ $[s]$', fontsize=20)
plt.show()

#plot MI
d_lays = [d*d_bin for d in range(-d_luft,d_luft)]

plt.figure(2,dpi=120)
plt.scatter(d_lays, MI_2_0, color='black')
plt.ylabel('$Mutual$ $information$ $I(n_2;n_0)$', fontsize=20)
plt.xlabel('$Poisson$ $spikes$ $delay$ $[s]$', fontsize=20)
plt.show()

