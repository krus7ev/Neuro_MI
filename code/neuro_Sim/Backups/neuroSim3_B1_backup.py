from __future__ import division

from neuro import *
from random import shuffle
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
#
#  4--->0     1<---5
#       ^     ^
#       |\1-g/|
#       | \ / |
#     g |  x  | g        Synapse strength g in [0,1]
#       | / \ |
#       |/   \|
#       2     3
#
# Poisson neurons {2,3}

mili  = 0.001          # Scaling factor 10^-3
# Global parameters ------------------------------------------------------------
E_l   = -70*mili       # Standard reverse potential
V_th  = -54*mili       # Vth = -40 [mV]
V_res = -80*mili       # Reset membrane potential
R_m   = 1              # Rm = 1[M_ohm]
I_e   = 18*mili        # Ie = 3.1[nA]
t_M   = 30*mili        # tau_m = 10[ms] = C_m*Rm time constant of the membrane
t_Ref = 5*mili         # Refractory period = 5[ms]
t_S   = 10*mili        # Time scale of the synapse
T     = 0.5            # total simulation period [s]
dt    = 1*mili         # Simulation time scale [ms]
bdt   = dt             # Bialek information estimation time scale
fsize = 25*mili        # Frame size

IF = 2                 # Number of Integrate and Fire neurons
PS  = 2                # Number of Poisson stimuli neurons
PN = 2                 # Number of Poisson noise neurons
N  = IF + PS + PN      # Total number of neurons

minRate = 10           # maximum average spike-rate for poisson neurons
maxRate = 50           # maximum average spike-rate for poisson neurons
rate_step = 10
rateRange = maxRate - minRate
noiseRate = 100

#samples = []                     # object storing experimental sample set
stimuli = int(rateRange/rate_step)   # number of stimuli (strengths)
repeats = 200                     # constant factor of minimum-times repetition
times = stimuli**(PS-1)*repeats   # repetition times = minimum * constant

g_res = 10             # no. of synapse strength steps over [0,1] ~> resolution
g_bin = 1/g_res        # width of synapse strength step based on g_res
g_E = 0.5
g_sqrd_E = 0
q = 0.1

h1 = h2 = 12           # size of open balls in spike-train metric spaces
tau_vR = 12*mili       # time-scale for the van-Rossum metric kernel

vR_metric = (metric('vR', tau = tau_vR), 'van Rossum')
VP_metric = (metric('VP', q = 166), 'Victor-Purpura')
metric = VP_metric

MI_1_0 = []
MI_1_0_E = 0
MI_1_0_sqrd_E = 0
MI_1_0xG_E = 0

MI_2_0 = [];       MI_2_0_B = []
MI_2_0_E = 0;      MI_2_0_B_E = 0
MI_2_0_sqrd_E = 0; MI_2_0_B_sqrd_E = 0
MI_2_0xG_E = 0;    MI_2_0_BxG_E = 0

MI_3_0 = [];       MI_3_0_B = []
MI_3_0_E = 0;      MI_3_0_B_E = 0
MI_3_0_sqrd_E = 0; MI_3_0_B_sqrd_E = 0
MI_3_0xG_E = 0;    MI_3_0_BxG_E = 0

MI_2_3 = []
MI_2_3_E = 0
MI_2_3_sqrd_E = 0
MI_2_3xG_E = 0

# Experiment Simulation---------------------------------------------------------
# Convey experiment over a variation of synapse strengths
for g in range(1, g_res) :
  g *= g_bin
  cMat = np.array([[  0,   0,  0,  0,  0,  0],
                   [  0,   0,  0,  0,  0,  0],
                   [  g, 1-g,  0,  0,  0,  0],
                   [1-g,   g,  0,  0,  0,  0],
                   [  q,   0,  0,  0,  0,  0],
                   [  0,   q,  0,  0,  0,  0]])

# create experiment object to store results from synapse-strength round
  sample = experiment()
# pre-generate the Poisson stimuli and combine their pairing
# in order to be able to distinguish them using Bialek's method
  pStimuli = []
  for i in range (PS) :
    pStimuli.append([])
  for s in range(stimuli) :
    sTs = -t_M + np.random.rand(PS)*t_M
    sRate = minRate + s*rate_step
    for p in range(PS) :
#     repeat each stimulus rate a number of times to ease Bialek prob. estimation
      for t in range(times):
        pStimuli[p] += [pNeuron(IF+p, sRate, T, dt, sTs[p])]

# TOTAL COMBINATORY ORDERING for any # of sources PS and valid "times" and "stimuli"
#   in each column we alternate stimuli**(PS-1-p)*repeats of each stimulus phase
#   (strength), initially all ordered with times = stimuli**(PS-1)*repeats of
#   each phase
  for p in range(1,PS) :
    t_size = stimuli**(PS-1-p)*repeats #current column's tuples size
    p_size = stimuli**(PS-p)*repeats   #parent column's tuples size
    for s in range(stimuli) :
       for t in range(times) :
          i = s*times + t
          mod_i = i % times
          _modi = i % t_size
          j = int(mod_i/t_size)*p_size + s*t_size + _modi
          pStimuli[p][i], pStimuli[p][j] = pStimuli[p][j], pStimuli[p][i]

# initialise and simulate network
  for s in range(stimuli) :
    for t in range(times) :
      neurons = []
      Vs = V_res + np.random.rand(IF)*(V_th - V_res)
      sTs = -t_M + np.random.rand(IF+PN)*t_M
      for i in range(IF) :
        neurons.append(neuron(i, Vs[i], sTs[i], E_l, V_th, V_res, R_m, I_e, t_M, t_Ref))
      for p in range(PS) :
        neurons.append(pStimuli[p][s*times + t])
      for n in range (PN) :
        neurons.append(pNeuron(IF+PS+n, noiseRate, T, dt, sTs[IF+n]))
#     Simulate network and save data
      sim = netSim(neurons, cMat, T, dt)
      Vs, Gs, raster = sim.simulate()
      sample.population.append(neurons) # no need for deepcopy
# add experiment results to samples set ? ANY USE
#  samples += [sample]
# assemble together spike trains elicited from each neuron over trials
  neuro_var = []
  for n in range(IF+PS):
    trains = []
    for st in range(stimuli*times):
      trains += [sample.population[st][n].sTrain]
    neuro_var.append(trains)

#-MI-BETWEEN-IF-NEURONS-0-AND-1
  MI_1_0_g = computeMI(neuro_var[1], neuro_var[0], metric[0], h1, h2)
  MI_1_0.append(MI_1_0_g)
  MI_1_0_E      += MI_1_0_g
  MI_1_0_sqrd_E += MI_1_0_g**2
  MI_1_0xG_E    += MI_1_0_g*g

#-MI-BETWEEN-POISSON-STIMULUS-2-AND-IF-NEURON-0
  MI_2_0_g = computeMI(neuro_var[2], neuro_var[0], metric[0], h1, h2)
  MI_2_0.append(MI_2_0_g)
  MI_2_0_E      += MI_2_0_g
  MI_2_0_sqrd_E += MI_2_0_g**2
  MI_2_0xG_E    += MI_2_0_g*g

#-USING-THE_BIALEK-METHOD-------------------------------------------------------
  MI_2_0_B_g = B_MI(neuro_var[2], 0, PS, stimuli, repeats, neuro_var[0], bdt, T, fsize)
  MI_2_0_B.append(MI_2_0_B_g)
  MI_2_0_B_E      += MI_2_0_B_g
  MI_2_0_B_sqrd_E += MI_2_0_B_g**2
  MI_2_0_BxG_E    += MI_2_0_B_g*g

#-MI-BETWEEN-POISSON-STIMULUS-3-AND-IF-NEURON-0
  MI_3_0_g = computeMI(neuro_var[3], neuro_var[0], metric[0], h1, h2)
  MI_3_0.append(MI_3_0_g)
  MI_3_0_E      += MI_3_0_g
  MI_3_0_sqrd_E += MI_3_0_g**2
  MI_3_0xG_E    += MI_3_0_g*g

#-USING-THE_BIALEK-METHOD-------------------------------------------------------
  MI_3_0_B_g = B_MI(neuro_var[3], 1, PS, stimuli, repeats, neuro_var[0], bdt, T, fsize)
  MI_3_0_B.append(MI_3_0_B_g)
  MI_3_0_B_E      += MI_3_0_B_g
  MI_3_0_B_sqrd_E += MI_3_0_B_g**2
  MI_3_0_BxG_E    += MI_3_0_B_g*g

#-MI-BETWEEN-POISSON-STIMULI-2-AND-3
  MI_2_3_g = computeMI(neuro_var[2], neuro_var[3], metric[0], h1, h2)
  MI_2_3.append(MI_2_3_g)
  MI_2_3_E      += MI_2_3_g
  MI_2_3_sqrd_E += MI_2_3_g**2
  MI_2_3xG_E    += MI_2_3_g*g

  g_sqrd_E += g**2

MI_1_0_E      /= (g_res-1)
MI_1_0_sqrd_E /= (g_res-1)
MI_1_0xG_E    /= (g_res-1)

MI_2_0_E        /= (g_res-1)
MI_2_0_sqrd_E   /= (g_res-1)
MI_2_0xG_E      /= (g_res-1)
MI_2_0_B_E      /= (g_res-1)
MI_2_0_B_sqrd_E /= (g_res-1)
MI_2_0_BxG_E    /= (g_res-1)

MI_3_0_E        /= (g_res-1)
MI_3_0_sqrd_E   /= (g_res-1)
MI_3_0xG_E      /= (g_res-1)
MI_3_0_B_E      /= (g_res-1)
MI_3_0_B_sqrd_E /= (g_res-1)
MI_3_0_BxG_E    /= (g_res-1)

MI_2_3_E      /= (g_res-1)
MI_2_3_sqrd_E /= (g_res-1)
MI_2_3xG_E    /= (g_res-1)

g_sqrd_E /= (g_res-1)

pearson_1_0 = (MI_1_0xG_E - MI_1_0_E*g_E)/np.sqrt((MI_1_0_sqrd_E - MI_1_0_E**2)*(g_sqrd_E - g_E**2))
pearson_2_0 = (MI_2_0xG_E - MI_2_0_E*g_E)/np.sqrt((MI_2_0_sqrd_E - MI_2_0_E**2)*(g_sqrd_E - g_E**2))
pearson_3_0 = (MI_3_0xG_E - MI_3_0_E*g_E)/np.sqrt((MI_3_0_sqrd_E - MI_3_0_E**2)*(g_sqrd_E - g_E**2))
pearson_2_3 = (MI_2_3xG_E - MI_2_3_E*g_E)/np.sqrt((MI_2_3_sqrd_E - MI_2_3_E**2)*(g_sqrd_E - g_E**2))

pearson_2_0_B = (MI_2_0_BxG_E - MI_2_0_B_E*g_E)/np.sqrt((MI_2_0_B_sqrd_E - MI_2_0_B_E**2)*(g_sqrd_E - g_E**2))
pearson_3_0_B = (MI_3_0_BxG_E - MI_3_0_B_E*g_E)/np.sqrt((MI_3_0_B_sqrd_E - MI_3_0_B_E**2)*(g_sqrd_E - g_E**2))

print 'Mean MI(n1;n0)          = ' + str(MI_1_0_E)
print 'Pearson coef. MI(n1;n0) = ' + str(pearson_1_0)

print 'Mean MI(n2;n0)          = ' + str(MI_2_0_E)
print 'using the Bialek method = ' + str(MI_2_0_B_E)
print 'Pearson coef. MI(n2;n0) = ' + str(pearson_2_0)
print 'using the Bialek method = ' + str(pearson_2_0_B)

print 'Mean MI(n3;n0)          = ' + str(MI_3_0_E)
print 'using the Bialek method = ' + str(MI_3_0_B_E)
print 'Pearson coef. MI(n3;n0) = ' + str(pearson_3_0)
print 'using the Bialek method = ' + str(pearson_3_0_B)

print 'Mean MI(n2;n3)          = ' + str(MI_2_3_E)
print 'Pearson coef. MI(n2;n3) = ' + str(pearson_2_3)

# Produce Graphs----------------------------------------------------------------
#set up x values for the conductance g
g_range  = [a*g_bin for a in range(1,g_res)]

#-MI(Neuron_2; Neuron_0)--------------------------------------------------------
#perform linear regression on generated data
y = np.array(MI_2_0)
x = np.array(g_range)
A = np.vstack([x, np.ones(len(x))]).T
m20, c20 = np.linalg.lstsq(A, y)[0]

#scatter Mutual Information between neurons 2 & 0
#and plot fitted line
plt.figure(1, dpi=120)
plt.scatter(g_range, MI_2_0, marker = '+', color = 'black',
            label='$MI_{2,0}$ computed with '+metric[1]+' metric'+
            '\n$g_{2,0}=g,$ mean $\\mu='+str(round(MI_2_0_E,2))+
            '$\ncorrelation coefficient $\\rho='+
            str(round(pearson_2_0,2))+'$')
plt.plot(x, m20*x + c20, linestyle='--', color='black',
         label='regression line: $I\\approx'+str(round(m20,2))+
         'g+'+str(round(c20,2))+'$')
plt.ylabel('$Mutual$ $information$ $I(n_2;n_0)$', fontsize=20)
plt.xlabel('$g$', fontsize=20)
#plt.ylim(-0.2,1.5)
plt.legend()
plt.show()

#-------------------------#
#-USING-THE-BIALEK-METHOD-#
#-------------------------#
y = np.array(MI_2_0_B)
m20, c20 = np.linalg.lstsq(A, y)[0]

plt.figure(12, dpi=120)
plt.scatter(g_range, MI_2_0_B, marker = '+', color = 'black',
            label='$MI_{2,0}$ computed using the Bialek Method'+
            '\n$g_{2,0}=g,$ mean $\\mu='+str(round(MI_2_0_B_E,2))+
            '$\ncorrelation coefficient $\\rho='+
            str(round(pearson_2_0_B,2))+'$')
plt.plot(x, m20*x + c20, linestyle='--', color='black',
         label='regression line: $I\\approx'+str(round(m20,2))+
         'g+'+str(round(c20,2))+'$')
plt.ylabel('$Mutual$ $information$ $I(n_2;n_0)$', fontsize=20)
plt.xlabel('$g$', fontsize=20)
#plt.ylim(-0.2,1.5)
plt.legend()
plt.show()


#-MI(Neuron_3; Neuron_0)--------------------------------------------------------
#perform linear regression
y = np.array(MI_3_0)
m30, c30 = np.linalg.lstsq(A, y)[0]

plt.figure(2, dpi=120)
plt.scatter(g_range, MI_3_0, marker = '+', color = 'black',
            label='$MI_{3,0}$ computed with '+metric[1]+' metric'+
            '\n$g_{3,0}=1-g,$ mean $\\mu='+str(round(MI_3_0_E,2))+
            '$\ncorrelation coefficient $\\rho='
            +str(round(pearson_3_0,2))+'$')
plt.plot(x, m30*x + c30, linestyle='--', color='black',
         label='regression line: $I\\approx'+str(round(m30,2))+
         'g+'+str(round(c30,2))+'$')
plt.ylabel('$Mutual$ $information$ $I(n_3;n_0)$', fontsize=20)
plt.xlabel('$g$', fontsize=20)
#plt.ylim(-0.2, 1.5)
plt.legend()
plt.show()

#-------------------------#
#-USING-THE-BIALEK-METHOD-#
#-------------------------#
y = np.array(MI_3_0_B)
m30, c30 = np.linalg.lstsq(A, y)[0]

plt.figure(22, dpi=120)
plt.scatter(g_range, MI_3_0_B, marker = '+', color = 'black',
            label='$MI_{2,0}$ computed using the Bialek Method'+
            '\n$g_{2,0}=1-g,$ mean $\\mu='+str(round(MI_3_0_B_E,2))+
            '$\ncorrelation coefficient $\\rho='+
            str(round(pearson_3_0_B,2))+'$')
plt.plot(x, m30*x + c30, linestyle='--', color='black',
         label='regression line: $I\\approx'+str(round(m30,2))+
         'g+'+str(round(c30,2))+'$')
plt.ylabel('$Mutual$ $information$ $I(n_3;n_0)$', fontsize=20)
plt.xlabel('$g$', fontsize=20)
#plt.ylim(-0.2,1.5)
plt.legend()
plt.show()


#-MI(Neuron_3; Neuron_0)-vs-MI(Neuron_2; Neuron_0)------------------------------
#perform linear regression
x = np.array(MI_2_0)
sortx = x.argsort()
x = np.array(x)[sortx]
A = np.vstack([x, np.ones(len(x))]).T
y = np.array(MI_3_0)
m3021, c3021 = np.linalg.lstsq(A, y)[0]

plt.figure(0, dpi=120)
plt.scatter(x, y, marker = '+', color = 'black',
            label='$MI_{3,0}$ vs $MI_{2,0}$, '+
            'both computed with '+metric[1]+' metric')
plt.plot(x, m3021*x + c3021, linestyle='--', color='black',
         label='regression line $MI_{3,0}\\approx'+str(round(m3021,2))+
         'MI_{2,0}+'+str(round(c3021,2))+'$')
plt.ylabel('$Mutual$ $information$ $I(n_3;n_0)$', fontsize=20)
plt.xlabel('$Mutual$ $information$ $I(n_2;n_0)$', fontsize=20)
#plt.ylim(-0.2, 1.5)
plt.legend()
plt.show()


#-MIXED-PAIRWISE-MI-PLOT--------------------------------------------------------
#perform linear regressions
y = np.array(MI_1_0)
m10, c10 = np.linalg.lstsq(A, y)[0]
y = np.array(MI_2_3)
m23, c23 = np.linalg.lstsq(A, y)[0]

plt.figure(3, dpi=120)

plt.scatter(g_range, MI_1_0, facecolors='none', edgecolors='red',
            label='$\\bar{I}(n_1;n_0)\\approx'+str(round(MI_1_0_E,2))+
            '$\ncorrelation coef. $\\rho_{1,0}='+str(round(pearson_1_0,2))+'$')
plt.plot(x, m10*x + c10, linestyle='--', color='red')

plt.scatter(g_range, MI_2_0, marker = '+', color = 'blue',
            label='$\\bar{I}(n_2;n_0)\\approx'+str(round(MI_2_0_E,2))+
            ',$ $g_{2,0}=g$'+
            '\ncorrelation coef. $\\rho_{2,0}='+str(round(pearson_2_0,2))+'$')
plt.plot(x, m20*x + c20, linestyle='--', color='blue')

plt.scatter(g_range, MI_3_0, marker = 'x', color = 'green',
            label='$\\bar{I}(n_3;n_0)\\approx'+str(round(MI_3_0_E,2))+
            ',$ $g_{3,0}=1-g$'+
            '\ncorrelation coef. $\\rho_{3,0}='+str(round(pearson_3_0,2))+'$')
plt.plot(x, m30*x + c30, linestyle='--', color='green')

plt.scatter(g_range, MI_2_3, facecolors='none', edgecolors='black',
            label='$\\bar{I}(n_2;n_3)\\approx'+str(round(MI_2_3_E,2))+
            '$\ncorrelation coef. $\\rho_{2,3}='+str(round(pearson_2_3,2))+'$')
plt.plot(x, m23*x + c23, linestyle='--', color='black')

plt.ylabel('$Mutual$ $information$', fontsize=20)
plt.xlabel('$g$', fontsize=20)
#plt.ylim(-0.2, 1.5)
plt.legend()
plt.show()
