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
# Integrate-and-fre neurons {0}
#
# 3--q-->0
#        ^
#        |\
#        | \
#      g |  \ g
#        |   \
#        |    \
#        1     2
#
# Poisson stimulus neurons {1,2}
# Poisson noise neurons {3}
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
T     = 0.6            # total simulation period [s]
dt    = 1*mili         # Simulation time scale [ms]
bdt   = 3*dt             # Bialek information estimation time scale
fsize = 30*mili        # Frame size

IF = 1                 # Number of Integrate and Fire neurons
PS = 2                 # Number of Poisson stimuli neurons
PN = 1                 # Number of Poisson noise neurons
N  = IF + PS + PN      # Total number of neurons

minRate_I  = 10
maxRate_I  = 40
rateStep_I = 10
sTims_I    = int((maxRate_I - minRate_I)/rateStep_I) + 1

minRate_J  = 25
maxRate_J  = 50
rateStep_J = 25
sTims_J    = int((maxRate_J - minRate_J)/rateStep_J) + 1

noiseRate = 100

g_e = 0.8
g_i = -g_e
q = 0.2

times  = 120
N_res  = [8, 10, 12, 15, 20, 24, 30, 40, 60]

h1 = h2 = 8           # size of open balls in spike-train metric spaces
tau_vR = 12*mili       # time-scale for the van-Rossum metric kernel

vR_metric = (metric('vR', tau = tau_vR), 'van Rossum')
VP_metric = (metric('VP', q = 166), 'Victor-Purpura')
metric = VP_metric

MI_2_0,   MI_2_0_B   = [], []  #2D: list of estimates at each averaging step
MI_2_0_E, MI_2_0_B_E = [], []  #1D: mean of MI estimates at each step

cMat = np.array([[  0,   0,  0,  0],
                 [g_e,   0,  0,  0],
                 [g_i,   0,  0,  0],
                 [  q,   0,  0,  0]])
                 
# Experiment Simulation---------------------------------------------------------
# Convey experiment over a variation of synapse strengths
for N_t in N_res :
  N_r = int(times/N_t)

  MI_2_0_nt,   MI_2_0_B_nt   = [], []
  MI_2_0_nt_E, MI_2_0_B_nt_E =  0,  0
  
  for r in range (N_r) :
    neuro_var = [ [] for i in range(IF+PS) ]
    stim_var  = [ [] for i in range(PS)    ]

    for si in range(sTims_I) :
      for sj in range(sTims_J) :
        rate_i = minRate_I + si*rateStep_I
        rate_j = minRate_J + sj*rateStep_J
        for t in range(N_t):
          neurons = []
          Vs = V_res + np.random.rand(IF)*(V_th - V_res)
          sTs = -t_M + np.random.rand(N)*t_M
  #       add IF neurons to simulation list
          for i in range(IF) :
            neurons += [neuron(i, Vs[i], sTs[i], E_l, V_th, V_res, R_m, I_e, t_M, t_Ref)]
  #       add the Poisson stimuli neurons combination
          neurons += [pNeuron(IF,   rate_i, T, dt, sTs[IF],   label = str(si)+str(sj))]
          neurons += [pNeuron(IF+1, rate_j, T, dt, sTs[IF+1], label = str(sj)+str(si))]
  #       add the Poisson noise input neurons
          for n in range (PN) :
            neurons += [pNeuron(IF+PS+n, noiseRate, T, dt, sTs[IF+PS+n])]
  #       Simulate network and save data
          sim = netSim(neurons, cMat, T, dt)
          Vs, Gs, raster = sim.simulate()
          for n_i in range(IF+PS):
             neuro_var[n_i].append(neurons[n_i].sTrain)
          for s_i in range(PS):
             stim_var[s_i].append(neurons[IF+s_i])

#---MI-BETWEEN-POISSON-STIMULUS-2-AND-IF-NEURON-0-------------------------------
#   compute the MI for estimation round; add to number/rounds averaging step set
    MI_2_0_r      = computeMI(neuro_var[2], neuro_var[0], metric[0], h1, h2)
    MI_2_0_nt    += [MI_2_0_r]
    MI_2_0_nt_E  += MI_2_0_r

#---USING-THE_BIALEK-METHOD
    MI_2_0_B_r     = B_MI(stim_var[0], neuro_var[0], bdt, T, fsize, N_t)
    MI_2_0_B_nt   += [MI_2_0_B_r]
    MI_2_0_B_nt_E += MI_2_0_B_r

  MI_2_0   += [MI_2_0_nt]
  MI_2_0_E += [MI_2_0_nt_E/N_r]
  
  MI_2_0_B   += [MI_2_0_B_nt]
  MI_2_0_B_E += [MI_2_0_B_nt_E/N_r]
  
MI_2_0_std   = []
MI_2_0_B_std = []

for nt in range(len(N_res)) :
  MI_2_0_std_nt   = 0 
  MI_2_0_B_std_nt = 0

  N_r = len(MI_2_0_B[nt])
  for r in range(N_r) :
    MI_2_0_std_nt   += (MI_2_0[nt][r]   - MI_2_0_E[nt]  )**2
    MI_2_0_B_std_nt += (MI_2_0_B[nt][r] - MI_2_0_B_E[nt])**2
  MI_2_0_std   += [ np.sqrt(MI_2_0_std_nt  /N_r) ]
  MI_2_0_B_std += [ np.sqrt(MI_2_0_B_std_nt/N_r) ]
    
  
# Produce Graphs----------------------------------------------------------------
# MI(R;S) Mean and std. dev. at each averaging step

plt.figure(1,dpi=120)
plt.errorbar(N_res, MI_2_0_E, yerr=MI_2_0_std, fmt='o', 
             label='$I(R;S)$ $computed$ $with$ \n$metric$-$space$ $method$ \n$N=$'
             + str(times))
plt.ylabel('$\langle I(S;R)_t \\rangle _r,$ $\sigma$', fontsize=20)
plt.xlabel('$t$\n$(\# trials$ $per$ $estimation$ $over$ $N/t$ $rounds)$', fontsize=20)
plt.xticks(N_res)
plt.xlim( (N_res[0]-1, N_res[-1]+1) )
plt.legend()
plt.show()



#USING BIALEK METHOD
plt.figure(2,dpi=120)
plt.errorbar(N_res, MI_2_0_B_E, yerr=MI_2_0_B_std, fmt='o', 
             label='$I(R;S)$ $computed$ $with$ \n$time$-$binning$ $method$ \n$N=$'
             + str(times))
plt.ylabel('$\langle I(S;R)_t \\rangle _r,$ $\sigma$', fontsize=20)
plt.xlabel('$t$\n$(\# trials$ $at$ $each$ $of$ $r=N/t$ $estimation$ $rounds)$', fontsize=20)
plt.xticks(N_res)
plt.xlim( (N_res[0]-1, N_res[-1]+1) )
plt.legend()
plt.show()


#BOTH ON SAME PLOT
#USING BIALEK METHOD
plt.figure(3,dpi=120)

plt.errorbar(N_res, MI_2_0_B_E, yerr=MI_2_0_B_std, fmt='s',
             label='$I(R;S)$ $computed$ $with$ \n$time$-$binning$ $method$ \n$N=$'
             + str(times))

plt.errorbar(N_res, MI_2_0_E, yerr=MI_2_0_std, fmt='o',
             label='$I(R;S)$ $computed$ $with$ \n$metric$-$space$ $method$ \n$N=$'
             + str(times))
                          
plt.ylabel('$\langle I(S;R)_t \\rangle _r,$ $\sigma$', fontsize=20)
plt.xlabel('$t$\n$(\# trials$ $at$ $each$ $of$ $r=N/t$ $estimation$ $rounds)$', fontsize=20)
plt.xticks(N_res)
plt.xlim( (N_res[0]-1, N_res[-1]+1) )
plt.legend()
plt.show()

