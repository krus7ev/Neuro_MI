from __future__ import division
import numpy as np
import scipy as scp
import matplotlib as mp
from pylab import *
import matplotlib.pyplot as plt


# ==============================================================================
# Object definitions
# ==============================================================================

# ------------------------------------------------------------------------------
# Neuron class
# ------------------------------------------------------------------------------
class neuron (object):    
  def __init__(self, idx, v0, st, el=None, vth=None, vres=None, rm=None, i=None, tm=None, tref=None, es=None):
    self.id    = idx        #index in connectivity matrix
    self.v     = v0         #Current voltage of the membrane    
    
    self.sTime  = st         #Time of last spike 
    self.sTrain = []
    
    self.e_l   = E_l   if el   == None else el      #Membrane revrse potential    
    self.v_th  = V_th  if vth  == None else vth     #Threshold potential
    self.v_res = V_res if vres == None else vres    #Reset potential  
    self.r_m   = R_m   if rm   == None else rm      #Resistance of the membrane
    self.i_ext = I_e   if i    == None else i       #External constant current
    self.t_m   = t_M   if tm   == None else tm      #Membrane time const=Cm*Rm
    self.t_ref = t_Ref if tref == None else tref    #Refractory period
    self.e_s   = 0     if es   == None else es      #Synapse reverse potential  


# returns value of f(V) = dV/dt-------------------------------------------------    
  def f(self, V, G_s):
    return (self.e_l - V + self.r_m*G_s*(self.e_s - V) + self.r_m*self.i_ext)/self.t_m   



# ------------------------------------------------------------------------------
# Synapse class
# ------------------------------------------------------------------------------   
class synapse:
  def __init__ (self, idx, ts=None) :
    self.id  = idx                        #index of post-synaptic neuron
    self.t_s = t_S if ts == None else ts   #time-scale of the synapse

    self.preSynTs = []      #list of times since pre-synaptic neurons spikes    
    self.preSynGs = []      #list of synaptic strengths/conductances
    self.G_s      = 0       #summed synaptic conductance

    
# Sum-up and update synaptic conductance----------------------------------------    
  def conduct(self):
    self.G_s = 0
    for i in range(len(self.preSynGs)):
      self.G_s += 0.5*self.preSynGs[i]*exp(-self.preSynTs[i]/self.t_s)
    return self.G_s



# ------------------------------------------------------------------------------
# Network class managing connectivty between neurons and synapses
# ------------------------------------------------------------------------------
class cnet (object):
  def __init__(self, cMat, nrns, syns):
    self.cMat = cMat
    
#   iterate j through neuron/synapse ids
    for j in range(len(syns)):
#     iterate i through indices of neurons pre-synaptic to neuron j 
      for i in ( self.list_pre_synapt_ns(syns[j]) ):
#       fill list of synaptic strengths/conductances 
        syns[j].preSynGs.append(cMat[i][j])
#       fill list of times since pre-synaptic neurons spikes 
        syns[j].preSynTs.append(0 - nrns[i].sTime)


# update list of times since pre-synaptic neuron spikes in synapse--------------
  def update_preSynTs(self, synapse, nrns, t):
#   iterate pre-synaptic neurons ids
    for j,k in enumerate( self.list_pre_synapt_ns(synapse) ):
      synapse.preSynTs[j] = t - nrns[k].sTime


# returns a list of post-synaptic neurons reached by a parameter neuron---------    
  def list_post_synapt_ns (self, neuron):
    idx = neuron.id
    return [item for sublist in np.nonzero(self.cMat[idx]) for item in sublist]


# returns a list of pre-synaptic neurons reaching a parameter synapse-----------  
  def list_pre_synapt_ns (self, synapse):
    idx = synapse.id
    return [item for sublist in np.nonzero(self.cMat[:,idx]) for item in sublist]


 
# ------------------------------------------------------------------------------
# Class simulating a network of neurons given:
# - a list of initial neuron voltags Vs
# - a list of initial neurons last spike times
# - a connectivity matrix cMat
# - a simulation period T
# - a time step dt
# passed as parameters, and using global simulation settings
# ------------------------------------------------------------------------------
class netSim :
  def __init__(self, Nrns, cMat, T, dt, h_t=None):    
    self.neurons  = Nrns               #list to store neuron objects
    self.synapses = []                 #list to store synapse objects
    self.cMat = cMat                   #connectivity matrix
    self.sts = sTs
    
    self.t  = np.arange(0, T, dt)      #time array
    self.dt = dt                       #time step 
    self.ht = 0 if h_t==None else h_t  #length of previous simulations (if any)

# Initialise synapse vector    
    for i in range(len(cMat)) :
      self.synapses.append(synapse(i))

#   2D np.array to store the simulation - neuron membrane potentials over time
    self.vSim   = np.zeros([len(self.neurons), len(self.t)])  
    self.gSim   = np.zeros([len(self.neurons), len(self.t)])
    self.raster = np.zeros([len(self.neurons), len(self.t)])*np.nan
      
#   create connectivity manager
    self.cNet = cnet(cMat, self.neurons, self.synapses)
        
    
# Ouptuts the membrane voltage at a single timeslice using RK4 appximation------
  def getV(self, nrn, t, G_s):
#   record spike and reset membrane potential
    if nrn.v >= nrn.v_th :
      nrn.v = nrn.v_res
        
#   keep reset value for a refracory period
    elif t < nrn.sTime + nrn.t_ref :
      nrn.v = nrn.v_res

#   inegreate        
    else :
      k1 = nrn.f(nrn.v, G_s )
      #k2 = nrn.f(nrn.v + k1*dt/2)
      #k3 = nrn.f(nrn.v + k2*dt/2)
      #k4 = nrn.f(nrn.v + k3*dt) 
      #nrn.v += 1/6*(k1 + 2*k2 + 2*k3 + k4)*dt
      nrn.v += k1*self.dt
    
    return nrn.v
 
   
# Simulate network--------------------------------------------------------------  
  def simulate(self):    
#   iterate i, t through time-array indices, steps
    for i, t in enumerate(self.t) :    
#     iterate j through neuron/synapse ids to update conductances
      for j in range(len(self.synapses)) :
#       update list of times since pre-synaptic neuron spikes in synapse j        
        self.cNet.update_preSynTs(self.synapses[j], self.neurons, t)
#       update conductance of synapse         
        self.gSim[j,i] = self.synapses[j].conduct()
          
#     iterate j through neuron ids to update memrane potentials
      for j in range(len(self.neurons)) :
#       update membrane potential of neuron j at time-id i  
        self.vSim[j,i] = self.getV(self.neurons[j], t, self.synapses[j].G_s)
                
#       check for spikes and record spike-timing       
        if self.neurons[j].v >= self.neurons[j].v_th :
          self.neurons[j].sTime = t          
          self.neurons[j].sTrain += [self.ht + t]
          self.raster[j][i] = j

#   reset sTime to minus-time-since-last-spike for use in subsequent simulations          
    for j in range(len(self.neurons)) :
      self.neurons[j].sTime = - (t - self.neurons[j].sTime)
                          
    return self.vSim, self.gSim, self.raster


# ------------------------------------------------------------------------------
# Functions implementing van Rossum metric
# ------------------------------------------------------------------------------

# filter spike train through exponential kernel to translate in vR form---------
def vR_mapTrain(spikes, dt, T, tau, mu) :
  time = np.arange(0, T, dt)
  h = np.zeros(len(time))
  
  for i, t_i in enumerate(time) :
    for j in range(len(spikes)) :
      t = t_i - spikes[j]
      h[i] += (1-mu)*exp(-t/tau) if t >= 0 else 0
      
  return h 

# calculate distance between spike trains in van Rossum form--------------------
def vR_computeDistance(f1, f2) :
  diff = np.zeros(len(f1))
  dist = 0
  for i in range(len(f1)) :
    diff[i] = f1[i] - f2[i]  
    dist += diff[i]*diff[i]*dt
    
  dist = sqrt(dist)

  return dist, diff

  
# ------------------------------------------------------------------------------
# Function implementing Victor-Purpura spike-time metric
# ------------------------------------------------------------------------------    
def VP_computeDistance(t1, t2, q) :
  G = np.zeros([len(t1), len(t2)]);
  G[:, 0] = [i for i in range(len(t1))]
  G[0, :] = [i for i in range(len(t2))]

  for i in range(1, len(t1)) :
    for j in range(1, len(t2)) :
      G[i][j] = min(G[i-1][j-1] + q*abs(t1[i]-t2[j]), G[i-1][j]+1, G[i][j-1]+1)
      
  return G[-1][-1]
      
      
      
# ==============================================================================
# MAIN SCRIPT       
# ==============================================================================    
    
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
  Neurons.append(neuron(i, Vs[i], sTs[i]))


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

dist_01_vR, diff_01_vR = vR_computeDistance(n0_vR, n1_vR);
print "dist_van_Rossum(nrn_0, nrn_1) = ", dist_01_vR


# Compute Victor Purpura distance-----------------------------------------------
q = 10
dist_01_VP = VP_computeDistance(Neurons[0].sTrain, Neurons[1].sTrain, q)

print "dist_VictorPurpura(nrn_0, nrn_1) = ", dist_01_VP



# Plot Graphs-------------------------------------------------------------------
'''
#plot voltages of neuron 1 & 2
figure(1)
p1, = plot(time, Vs[0], 'g')  
p2, = plot(time, Vs[1], 'r')
plt.xlabel('$t$ $[s]$', fontsize=15)
plt.ylabel('$V_m$', fontsize=15)
show()


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

