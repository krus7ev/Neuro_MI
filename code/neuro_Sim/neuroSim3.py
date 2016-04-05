from __future__ import division
import numpy as np
import scipy as scp
import matplotlib as mp
import matplotlib.pyplot as plt
from copy import deepcopy

from pylab import *
from neuro import *

plt.switch_backend('QT4Agg')
    
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

minRate = 20           # maximum average spike-rate for poisson neurons
maxRate = 30           # maximum average spike-rate for poisson neurons
rateRange = maxRate - minRate

data = []
trials  = 30           # number of simulation rounds in experiment

delay  = 0.125
d_luft = 50
d_bin  = 10**(-len(str(d_luft)))

h1 = 10                # size of open balls in spike-trrain metric space
h2 = 10

tau_vR = 1
vR_metric = metric('vR', dt=dt, T=T, tau=tau_vR, mu=0)
VP_metric = metric('VP', q=200)

metric = vR_metric

MI_2_0 = []
MI_2_0_Sum = 0

MI_2_1 = []
MI_2_1_Sum = 0

g = 0.8               # synapse 2-0 and 3-1 strength == 1 minus 2-1 or 3-0

# Experiments Simulation--------------------------------------------------------
#set up network topology
sample = experiment()
cMat = np.array([[  0,   0,  0,  0],
                 [  0,   0,  0,  0],
                 [  g, 1-g,  0,  0],
                 [1-g,   g,  0,  0]])

# simulate network a number of trials to generate sample data
for r in range(trials) :
# Initialise network parameters
  Vs = V_res + np.random.rand(IF)*(V_th - V_res)
  sTs = -t_M + np.random.rand(N)*t_M
  sRates = minRate + np.random.rand(P)*(rateRange)
  
  neurons = []
  for i in range(IF) :
    neurons.append(neuron(i, Vs[i], sTs[i], E_l, V_th, V_res, R_m, I_e, t_M, t_Ref))
  for i in range(IF, N) :
    pN = pNeuron(i, sRates[i-IF], T, dt, sTs[i])
    pN.delay(delay)
    neurons.append(pN)

# Simulate network and save data
  sample.simulation += [netSim(neurons, cMat, T, dt)]
  Vs, Gs, raster = sample.simulation[-1].simulate()

  sample.population += [neurons]
#  sample.VsResults  += [Vs]
#  sample.GsResults  += [Gs]
#  sample.rasters    += [raster]

#assemble together spike trains elicited from each neuron over trials
neuro_var = []
for n in range(IF):
  s_trains = []
  for r in range(trials):
    s_trains += [sample.population[r][n].sTrain]
  neuro_var += [s_trains]
for n in range(IF, N):
  neuro_var += [[]]

# Compute mutual information sliding trhough different delays
for d in range(-d_luft, d_luft) :
  d *= d_bin
# add delay to poisson trains
  for n in range(IF, N):
    s_trains = []
    for r in range(trials):
      pN = deepcopy(sample.population[r][n])
      pN.delay(d)
      s_trains += [pN.sTrain]
    neuro_var[n] = s_trains

  MI_2_0 += [computeMI(neuro_var[2], neuro_var[0], metric, h1, h2)]
  MI_2_0_Sum += MI_2_0[-1]
  
#  MI_2_1 += [computeMI(neuro_var[2], neuro_var[1], metric, h1, h2)]
#  MI_2_1_Sum += MI_2_1[-1]

MI_2_0_Avg = MI_2_0_Sum / (2*d_luft)
#MI_2_1_Avg = MI_2_1_Sum / (alpha_res-1)

#print 'Average MI(n2;n0) = ' + str(MI_2_0_Avg)
#print 'Average MI(n2;n1) = ' + str(MI_2_1_Avg)
#e.g.:
#Average MI(n2;n0) = 0.826254919525
#Average MI(n2;n1) = 0.766732206785



# Produce Graphs----------------------------------------------------------------

#plot raster
time = np.arange(0, T, dt)

figure(1)
ylim([-1, N])
yticks(np.arange( 0, N, 1))
xticks(np.arange(0, T, 0.05))
p3 = plot(time, np.transpose(raster), 'b.')
plt.grid()
plt.ylabel('$Neuron$ $indices$', fontsize=20)
plt.xlabel('$Spike$ $times$ $[s]$', fontsize=20)
show()

#plot MI
d_lays = [d*d_bin for d in range(-d_luft,d_luft)]
#d_lays_ = [1-d for d in d_lays]
figure(2)
#plot(d_lays, alpha) 
scatter(d_lays, MI_2_0)
show()

'''
figure(3)
plot(alpha, alpha_)
scatter(alpha, MI_2_1)
show()
'''
