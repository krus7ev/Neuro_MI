from __future__ import division

from neuro import *

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
maxRate = 50           # maximum average spike-rate for poisson neurons
rateRange = maxRate - minRate

samples = []           # object storing experimental sample set
data = []
trials  = 30           # number of simulation rounds in experiment

g_res = 30             # no. of synapse strength steps over [0,1] ~> resolution
g_bin = 1.0/g_res      # width of synapse strength step based on g_res

h1 = 10                # size of open balls in spike-trrain metric space
h2 = 10

tau_vR = 12*mili
vR_metric = metric('vR', tau = tau_vR)
VP_metric = metric('VP', q=200)

#metric = vR_metric
metric = VP_metric

MI_2_0 = []
MI_2_0_Sum = 0

MI_2_1 = []
MI_2_1_Sum = 0


# Experiment Simulation---------------------------------------------------------

# Convey experiment over a variation of synapse strengths
for g in range(1, g_res) :
  g *= g_bin
  sample = experiment()
  cMat = np.array([[  0,   0,  0,  0],
                   [  0,   0,  0,  0],
                   [  g, 1-g,  0,  0],
                   [1-g,   g,  0,  0]])

# simulate network a number of trials to generate sample data
  for r in range(trials) :
#   Initialise network parameters
    Vs = V_res + np.random.rand(IF)*(V_th - V_res)
    sTs = -t_M + np.random.rand(N)*t_M
    sRates = minRate + np.random.rand(P)*(rateRange)

    neurons = []
    for i in range(IF) :
      neurons.append(neuron(i, Vs[i], sTs[i], E_l, V_th, V_res, R_m, I_e, t_M, t_Ref))
    for i in range(IF, N) :
      neurons.append(pNeuron(i, sRates[i-IF], T, dt, sTs[i]))

#   Simulate network and save data
    sample.simulation += [netSim(neurons, cMat, T, dt)]
    Vs, Gs, raster = sample.simulation[-1].simulate()

    sample.population += [neurons]
#    sample.VsResults  += [Vs]
#    sample.GsResults  += [Gs]
#    sample.rasters    += [raster]

# add experiment results to samples set    
  samples += [sample]
 
# assemble together spike trains elicited from each neuron over trials
  neuro_var = []
  for n in range(N):
    s_trains = []
    for r in range(trials):
      s_trains += [samples[-1].population[r][n].sTrain]
    
    neuro_var += [s_trains]
   
  MI_2_0 += [computeMI(neuro_var[2], neuro_var[0], metric, h1, h2)]
  MI_2_0_Sum += MI_2_0[-1]
  
#  MI_2_1 += [computeMI(neuro_var[2], neuro_var[1], metric, h1, h2)]
#  MI_2_1_Sum += MI_2_1[-1]

MI_2_0_Avg = MI_2_0_Sum / (g_res-1)
#MI_2_1_Avg = MI_2_1_Sum / (g_res-1)

print 'Average MI(n2;n0) = ' + str(MI_2_0_Avg)
#print 'Average MI(n2;n1) = ' + str(MI_2_1_Avg)
#e.g.:
#Average MI(n2;n0) = 0.826254919525
#Average MI(n2;n1) = 0.766732206785



# Produce Graphs----------------------------------------------------------------

# Plot Mutual Information

#set up x values (g - for conductance)
g_range  = [a*g_bin for a in range(1,g_res)]
#use this to plot a line with slope -1 descending from y=1 till y=0
g_range_ = [1-g for g in g_range]

#perform linear regression on generated data
y = np.array(MI_2_0)
x = np.array(g_range)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]

#scatter Mutual Information between neurons 0 & 1 
#and plot fitted line
plt.figure(2)
plt.plot(x, m*x + c)
plt.plot(x, x) 
plt.scatter(x, y)
plt.ylabel('$Mutual$ $information$ $I(n_2;n_0)$', fontsize=20)
plt.xlabel('$Synapse$ $strength$ $g_{2,0}$', fontsize=20)
plt.show()

'''
plt.figure(3)
plt.plot(g_range, g_range_)
plt.scatter(g_range, MI_2_1)
plt.ylabel('$Mutual$ $information$ $I(n_2;n_1)$', fontsize=20)
plt.xlabel('$Synapse$ $strength$ $g_{2,1}$', fontsize=20)
plt.show()
'''

# Check out latest raster plot
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
