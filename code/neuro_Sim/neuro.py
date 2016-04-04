from __future__ import division
import numpy as np
import math
import scipy as scp
import matplotlib as mp
from pylab import *
import matplotlib.pyplot as plt

# ==============================================================================
# Global definitions
# ==============================================================================
mili  = 0.001          # Scaling factor 10^-3
t_S   = 10*mili        # Time scale of the synapse


# ==============================================================================
# Object definitions
# ==============================================================================

# ------------------------------------------------------------------------------
# IF Neuron class
# ------------------------------------------------------------------------------
class neuron (object):    
  def __init__(self, idx, v0, st, el=None, vth=None, vres=None, rm=None, i=None, tm=None, tref=None, es=None):
    self.id    = idx        #index in connectivity matrix
    self.type  = 'IF'
    self.v     = v0         #Current voltage of the membrane    
    
    self.sTime  = st        #Time of last spike 
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
  def __init__ (self, idx, dt, ts=None) :
    self.id  = idx                         #index of post-synaptic neuron
    self.t_s = t_S if ts == None else ts   #time-scale of the synapse
    self.dt = dt
    
    self.preSynTs = []        #list of times since pre-synaptic neurons spikes    
    self.preSynGs = []        #list of synaptic strengths/conductances
    self.G_s_1      = 0       #summed synaptic conductance
    self.G_s_23     = 0
    self.G_s_4      = 0

# Sum-up and update synaptic conductance----------------------------------------    
  def conduct(self):
    self.G_s_1      = 0
    self.G_s_23     = 0
    self.G_s_4      = 0
    for i in range(len(self.preSynGs)):
      self.G_s_1  += 0.5*self.preSynGs[i]*exp(- self.preSynTs[i] / self.t_s)
      self.G_s_23 += 0.5*self.preSynGs[i]*exp(-(self.preSynTs[i] + 0.25*self.dt)/self.t_s)
      self.G_s_4  += 0.5*self.preSynGs[i]*exp(-(self.preSynTs[i] + self.dt)/self.t_s)
    return self.G_s_1



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
    self.allNrns  = Nrns
    self.neurons  = []
    self.poissons = []
    self.synapses = []
    self.cMat     = cMat               #connectivity matrix   

    self.t  = np.arange(0, T, dt)      #time array
    self.dt = dt                       #time step 
    self.ht = 0 if h_t==None else h_t  #length of previous simulations (if any)
    
#   Add neuron and synapse objects only for Integrate-and-Fire neurons
    for i in range(len(self.allNrns)) :
      if self.allNrns[i].type == 'IF':
        self.neurons.append(self.allNrns[i])
        self.synapses.append(synapse(i, self.dt))
      elif Nrns[i].type == 'P':
        self.poissons.append(self.allNrns[i])

#   2D np.array to store the simulation - neuron membrane potentials over time
    self.vSim   = np.zeros([len(self.neurons), len(self.t)])  
    self.gSim   = np.zeros([len(self.neurons), len(self.t)])
    self.raster = np.zeros([len(self.allNrns), len(self.t)])*np.nan
      
#   create connectivity manager
    self.cNet = cnet(cMat, self.allNrns, self.synapses)
        
    
# Ouptuts the membrane voltage at a single timeslice using RK4 appximation------
  def getV(self, nrn, t, G_s_1, G_s_23, G_s_4):
#   record spike and reset membrane potential
    if nrn.v >= nrn.v_th :
      nrn.v = nrn.v_res
        
#   keep reset value for a refracory period
    elif t < nrn.sTime + nrn.t_ref :
      nrn.v = nrn.v_res

#   inegreate        
    else :
      k1 = self.dt*nrn.f(nrn.v, G_s_1 )
      k2 = self.dt*nrn.f(nrn.v + k1/2, G_s_23)
      k3 = self.dt*nrn.f(nrn.v + k2/2, G_s_23)
      k4 = self.dt*nrn.f(nrn.v + k3, G_s_4) 
      nrn.v += 1/6*(k1 + 2*k2 + 2*k3 + k4)
    
    return nrn.v
 
   
# Simulate network--------------------------------------------------------------  
  def simulate(self):    
#   iterate i, t through time-array indices, steps
    for i, t in enumerate(self.t) :    
#     iterate j through neuron/synapse ids to update synaptic conductances
      for j in range(len(self.synapses)) :
#       update times since pre-synaptic spikes in synapse j; conduct         
        self.cNet.update_preSynTs(self.synapses[j], self.allNrns, t)        
        self.gSim[j,i] = self.synapses[j].conduct()
          
#     iterate j through neuron ids to update memrane potentials
      for j in range(len(self.neurons)) :
#       update membrane potential of neuron j at time-id i  
        self.vSim[j,i] = self.getV( self.neurons[j], t, 
                                    self.synapses[j].G_s_1, 
                                    self.synapses[j].G_s_23, 
                                    self.synapses[j].G_s_4 )
                
#       check for spikes and record spike-timing       
        if self.neurons[j].v >= self.neurons[j].v_th :
          self.neurons[j].sTime = t          
          self.neurons[j].sTrain += [self.ht + t]
          self.raster[j][i] = j
          
#     check if poisson spike has occurred
      for p in range(len(self.poissons)):
        if t in self.poissons[p].sTrain :
          self.poissons[p].sTime = t
          self.raster[self.poissons[p].id][i] = self.poissons[p].id 
          

#   reset sTime to minus-time-since-last-spike for use in subsequent simulations          
    for j in range(len(self.allNrns)) :
      self.allNrns[j].sTime = - (t - self.allNrns[j].sTime)
                          
    return self.vSim, self.gSim, self.raster



# ------------------------------------------------------------------------------
# Class generating Poisson neuron
# ------------------------------------------------------------------------------
class pNeuron (object):    
  def __init__(self, idx, sRate, T, dt, st):
    self.id     = idx            #index in connectivity matrix
    self.type   = 'P'
    self.sTrain = []             #spike times
    self.sTime  = st             #time of last spike
    self.count  = 0              #spike count over period T
    
    time = np.arange(0, T, dt)
    for i, t in enumerate(time) :
      x = np.random.rand()
      if x < sRate*dt :
        self.sTrain += [t]
        self.count += 1



# ------------------------------------------------------------------------------
# Function generating Poisson spike trains given a mean rate (FOR TESTING)
# ------------------------------------------------------------------------------      
def getPoissonTrain(T, dt, sRate) :
  train = []
  time = np.arange(0, T, dt)
  count = 0
    
  for i, t in enumerate(time) :
    x = np.random.rand()
    if x < sRate*dt :
      train += [t]
      count += 1
      
  return train, count



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
      h[i] += (1-mu)*tau*exp(-t/tau) if t >= 0 else 0
      
  return h 

# calculate distance between spike trains in van Rossum form--------------------
def vR_computeDistance(f1, f2, dt) :
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
# safety conditions
  if len(t1) == 0 and len(t2) == 0:
    return 1000
  if len(t1) == 0 :
    return len(t2)
  if len(t2) == 0 : 
    return len(t1)

  G = np.zeros([len(t1), len(t2)])
  G[:, 0] = [i for i in range(len(t1))]
  G[0, :] = [i for i in range(len(t2))]

  for i in range(1, len(t1)) :
    for j in range(1, len(t2)) :
      G[i][j] = min(G[i-1][j-1] + q*abs(t1[i]-t2[j]), G[i-1][j]+1, G[i][j-1]+1)
      
  return G[-1][-1]




# ------------------------------------------------------------------------------
# objects to store the settings for a metric
# ------------------------------------------------------------------------------
class metric(object):
  def __init__(self, name, dt=None, T=None, tau=None, mu=None, q=None):
    self.type = name
    
    if name == 'vR':
      self.dt  = dt
      self.T   = T
      self.tau = tau
      self.mu  = mu
      
    elif name == 'VP':
      self.q  = q




#-------------------------------------------------------------------------------
# Compute distance matrix given a set of spike trains and a metric
#-------------------------------------------------------------------------------
def getDistanceMap(trains, metric):
  N = len(trains)
  dMap = np.zeros([N,N])
  
  if metric.type == 'vR':  
    vR_trains = []
    for t in range(N):
      vR_trains += [vR_mapTrain(trains[t], metric.dt, metric.T, metric.tau, metric.mu)]
    
  for i in range(N):
    for j in range(N):
      if dMap[j][i] != 0 :
        dMap[i][j] = dMap[j][i]
      elif i != j :      
        if metric.type == 'vR':
          dMap[i][j], dif = vR_computeDistance(vR_trains[i], vR_trains[j], metric.dt)
        elif metric.type == 'VP':
          dMap[i][j] = VP_computeDistance(trains[i], trains[j], metric.q)
  
  return dMap

  
  
# ------------------------------------------------------------------------------
# Class to store all components of a simulation experiment
# ------------------------------------------------------------------------------  
class experiment(object):
  def __init__(self):
    self.simulation = []      #simulation obejcts of all trials
    self.population = []      #populations of neuron objects of all trials
    self.VsResults  = []      #2D voltages-over-time arrays of all trials
    self.GsResults  = []      #2D conductances-over-time arrays of all trials
    self.rasters    = []      #raster plots of all trials


#-------------------------------------------------------------------------------
# Compute mutual information between two sets of spike trains
# given a metric and a resolution for the model
#-------------------------------------------------------------------------------
def computeMI(S, R, metric, h1, h2) :
  N = len(S)
  if h1 >= N or h2 >=N:
    raise Exception("h1 or h2 >= N!")
  
  h1h2 = h1*h2  #+1 like Laplace - (s,r) is counted in the ball  
  MI = 0

  dMap_S = getDistanceMap(S, metric)
  dMap_R = getDistanceMap(R, metric)
  
  for i in range(N):
    b_si = dMap_S[i].argsort()[:h1+1] 
    b_ri = dMap_R[i].argsort()[:h2+1]
                                            # DO NOT:
    count = len(np.intersect1d(b_si, b_ri)) #- 1 subtract the actual entry (s,r)
    if count <= 0: #just in case
      print "Warning: count<=0 --> set count=1!"
      count = 1
      
    MI += math.log((N*count)/h1h2, 2)
  

  MI = MI/N

  return MI
