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
trials  = 48           # number of simulation rounds in experiment

g_res = 50             # no. of synapse strength steps over [0,1] ~> resolution
g_bin = 1.0/g_res      # width of synapse strength step based on g_res
g_E = 0.5
g_sqrd_E = 0


h1 = 12                # size of open balls in spike-trrain metric space
h2 = 12

tau_vR = 12*mili
vR_metric = (metric('vR', tau = tau_vR), 'van Rossum')
VP_metric = (metric('VP', q = 166), 'Victor-Purpura')

#metric = vR_metric
metric = vR_metric

MI_1_0 = []
MI_1_0_E = 0
MI_1_0_sqrd_E = 0
MI_1_0xG_E = 0 # mean of MI_1_0 x g

MI_2_0 = []
MI_2_0_E = 0
MI_2_0_sqrd_E = 0
MI_2_0xG_E = 0 # mean of MI_2_0 x g

MI_3_0 = []
MI_3_0_E = 0
MI_3_0_sqrd_E = 0
MI_3_0xG_E = 0 # mean of MI_3_0 x g

MI_2_3 = []
MI_2_3_E = 0
MI_2_3_sqrd_E = 0
MI_2_3xG_E = 0 # mean of MI_2_3 x g


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

# add experiment results to samples set    
  samples += [sample]
 
# assemble together spike trains elicited from each neuron over trials
  neuro_var = []
  for n in range(N):
    s_trains = []
    for r in range(trials):
      s_trains += [samples[-1].population[r][n].sTrain]
    
    neuro_var += [s_trains]
  
  MI_1_0_g = computeMI(neuro_var[1], neuro_var[0], metric[0], h1, h2) 
  MI_1_0.append(MI_1_0_g)
  MI_1_0_E      += MI_1_0_g
  MI_1_0_sqrd_E += MI_1_0_g**2
  MI_1_0xG_E    += MI_1_0_g*g
  
  MI_2_0_g = computeMI(neuro_var[2], neuro_var[0], metric[0], h1, h2) 
  MI_2_0.append(MI_2_0_g)
  MI_2_0_E      += MI_2_0_g
  MI_2_0_sqrd_E += MI_2_0_g**2
  MI_2_0xG_E    += MI_2_0_g*g 
  
  MI_3_0_g = computeMI(neuro_var[3], neuro_var[0], metric[0], h1, h2)
  MI_3_0.append(MI_3_0_g)
  MI_3_0_E      += MI_3_0_g
  MI_3_0_sqrd_E += MI_3_0_g**2
  MI_3_0xG_E    += MI_3_0_g*g
  
  MI_2_3_g = computeMI(neuro_var[2], neuro_var[3], metric[0], h1, h2) 
  MI_2_3.append(MI_2_3_g)
  MI_2_3_E      += MI_2_3_g
  MI_2_3_sqrd_E += MI_2_3_g**2
  MI_2_3xG_E    += MI_2_3_g*g
  
  g_sqrd_E += g**2

MI_1_0_E      /= (g_res-1)
MI_1_0_sqrd_E /= (g_res-1)
MI_1_0xG_E    /= (g_res-1)

MI_2_0_E      /= (g_res-1)
MI_2_0_sqrd_E /= (g_res-1)
MI_2_0xG_E    /= (g_res-1)

MI_3_0_E      /= (g_res-1)
MI_3_0_sqrd_E /= (g_res-1)
MI_3_0xG_E    /= (g_res-1)

MI_2_3_E      /= (g_res-1)
MI_2_3_sqrd_E /= (g_res-1)
MI_2_3xG_E    /= (g_res-1)

g_sqrd_E /= (g_res-1)

pearson_1_0 = (MI_1_0xG_E - MI_1_0_E*g_E)/np.sqrt((MI_1_0_sqrd_E - MI_1_0_E**2)*(g_sqrd_E - g_E**2))
pearson_2_0 = (MI_2_0xG_E - MI_2_0_E*g_E)/np.sqrt((MI_2_0_sqrd_E - MI_2_0_E**2)*(g_sqrd_E - g_E**2))
pearson_3_0 = (MI_3_0xG_E - MI_3_0_E*g_E)/np.sqrt((MI_3_0_sqrd_E - MI_3_0_E**2)*(g_sqrd_E - g_E**2))
pearson_2_3 = (MI_2_3xG_E - MI_2_3_E*g_E)/np.sqrt((MI_2_3_sqrd_E - MI_2_3_E**2)*(g_sqrd_E - g_E**2))

print 'Mean MI(n1;n0)          = ' + str(MI_1_0_E)
print 'Pearson coef. MI(n1;n0) = ' + str(pearson_1_0)
print 'Mean MI(n2;n0)          = ' + str(MI_2_0_E)
print 'Pearson coef. MI(n2;n0) = ' + str(pearson_2_0)
print 'Average MI(n3;n0)       = ' + str(MI_3_0_E)
print 'Pearson coef. MI(n3;n0) = ' + str(pearson_3_0)
print 'Mean MI(n2;n3)          = ' + str(MI_2_3_E)
print 'Pearson coef. MI(n2;n3) = ' + str(pearson_2_3)


# Produce Graphs----------------------------------------------------------------
# Plot Mutual Information

#set up x values (g - for conductance)
g_range  = [a*g_bin for a in range(1,g_res)]

# MI(Neuron_2; Neuron_0)
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
plt.ylim(-0.2,1.5)
plt.legend()
plt.show()


# MI(Neuron_2; Neuron_0)
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
plt.ylim(-0.2, 1.5)
plt.legend()
plt.show()


# MI(Neuron_3; Neuron_0) vs MI(Neuron_2; Neuron_0)
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
plt.ylim(-0.2, 1.5)
plt.legend()
plt.show()


# MIXED PAIRWISE MI PLOT
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
plt.ylim(-0.2, 1.5)
plt.legend()
plt.show()

