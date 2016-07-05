from __future__ import division
import numpy as np
import math
import scipy as scp
import matplotlib as mp
from math import exp
#from pylab import *
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
  def __init__(self, idx, v0, st, el=None, vth=None, vres=None, rm=None,
               i=None, tm=None, tref=None, es=None):
    self.id     = idx                               #id in connectivity matrix
    self.type   = 'IF'                              #type of neuron
    self.v      = v0                                #membrane potential
    self.sTime  = st                                #time of last spike
    self.sTrain = []                                #spike times of the neuron
    self.e_l    = E_l   if el   == None else el     #membrane reverse potential
    self.v_th   = V_th  if vth  == None else vth    #threshold potential
    self.v_res  = V_res if vres == None else vres   #reset potential
    self.r_m    = R_m   if rm   == None else rm     #membrane resistance
    self.i      = I_e   if i    == None else i      #constant input current
    self.t_m    = t_M   if tm   == None else tm     #membrane time constant
    self.t_ref  = t_Ref if tref == None else tref   #refractory period
    self.e_s    = 0     if es   == None else es     #synapse reverse potential

# returns value of f(V) = dV/dt
  def f(self, V, G_s):
    f = (self.e_l - V + self.r_m*G_s*(self.e_s - V) + self.r_m*self.i)/self.t_m
    return f


# ------------------------------------------------------------------------------
# Synapse class
# ------------------------------------------------------------------------------
class synapse (object):
  def __init__ (self, idx, dt, ts=None) :
    self.id     = idx                         #index of post-synaptic neuron
    self.t_s    = t_S if ts == None else ts   #time-scale of the synapse
    self.dt     = dt
    self.sTs   = []                           #times since pre-synaptic spikes
    self.sGs   = []                           #synaptic strengths
    self.Gs_1  = 0                           #summed synaptic conductance
    self.Gs_23 = 0
    self.Gs_4  = 0

# Sum-up and update synaptic conductance
  def conduct(self):
    self.Gs_1  = 0
    self.Gs_23 = 0
    self.Gs_4  = 0
    for i in range(len(self.sGs)):
      self.Gs_1 +=0.5*self.sGs[i]*exp(- self.sTs[i]/self.t_s)
      self.Gs_23+=0.5*self.sGs[i]*exp(-(self.sTs[i]+0.25*self.dt)/self.t_s)
      self.Gs_4 +=0.5*self.sGs[i]*exp(-(self.sTs[i]+self.dt)/self.t_s)
    return self.Gs_1


# ------------------------------------------------------------------------------
# Network class managing connectivty between neurons and synapses
# ------------------------------------------------------------------------------
class cnet (object):
  def __init__(self, cMat, nrns, syns):
    self.cMat = cMat
#   iterate through neuron/synapse ids
    for j in range(len(syns)):
#     iterate through indices of neurons pre-synaptic to neuron j
      for i in ( self.list_pre_synapt_ns(syns[j]) ):
#       fill list of synaptic strengths and pre-synaptic spikes
        syns[j].sGs.append(cMat[i][j])
        syns[j].sTs.append(0 - nrns[i].sTime)

# update times since pre-synaptic spikes in synapse
  def update_sTs(self, synapse, nrns, t):
#   iterate pre-synaptic neurons ids
    for j,k in enumerate( self.list_pre_synapt_ns(synapse) ):
      synapse.sTs[j] = t - nrns[k].sTime

# return pre-synaptic neurons reaching parameter synapse
  def list_pre_synapt_ns (self, synapse):
    cMat = self.cMat; id = synapse.id
    return [item for sublist in np.nonzero(cMat[:,id]) for item in sublist]


# ------------------------------------------------------------------------------
# Class simulating a network of neurons given:
# - a list of initial neuron voltags Vs
# - a list of initial neurons last spike times
# - a connectivity matrix cMat
# - a simulation period T
# - a time step dt
# passed as parameters, and using global simulation settings
# ------------------------------------------------------------------------------
class netSim (object) :
  def __init__(self, Nrns, cMat, T, dt, h_t=None):
    self.allNrns  = Nrns
    self.neurons  = []
    self.poissons = []
    self.synapses = []
    self.cMat     = cMat                     #connectivity matrix
    self.t        = np.arange(0, T, dt)      #time array
    self.dt       = dt                       #time step
    self.ht       = 0 if h_t==None else h_t  #chronologic time incurred
    for i in range(len(self.allNrns)) :      #neurons and synapse objects
      if self.allNrns[i].type == 'IF':
        self.neurons.append(self.allNrns[i])
        self.synapses.append(synapse(i, self.dt))
      elif Nrns[i].type == 'P':
        self.poissons.append(self.allNrns[i])
    self.cNet   = cnet(cMat, self.allNrns, self.synapses)
#   arrays storing simulation data - potential and conductivity over time
    self.vSim   = np.zeros([len(self.neurons), len(self.t)])
    self.gSim   = np.zeros([len(self.neurons), len(self.t)])
    self.raster = np.zeros([len(self.allNrns), len(self.t)])*np.nan

# Compute membrane potential at a single timeslice using RK4 appximation
  def getV(self, nrn, t, Gs_1, Gs_23, Gs_4):
    if nrn.v >= nrn.v_th :             #record spikes and reset potential
      nrn.v = nrn.v_res
    elif t < nrn.sTime + nrn.t_ref :   #hold reset if refracory period
      nrn.v = nrn.v_res
    else :                             #inegreate using RK4 method
      k1 = self.dt*nrn.f(nrn.v, Gs_1 )
      k2 = self.dt*nrn.f(nrn.v + k1/2, Gs_23)
      k3 = self.dt*nrn.f(nrn.v + k2/2, Gs_23)
      k4 = self.dt*nrn.f(nrn.v + k3, Gs_4)
      nrn.v += 1/6*(k1 + 2*k2 + 2*k3 + k4)
    return nrn.v

# Simulate network
  def simulate(self):
    for i, t in enumerate(self.t) :
      for j in range(len(self.synapses)) :             #update conductances
        self.cNet.update_sTs(self.synapses[j], self.allNrns, t)
        self.gSim[j,i] = self.synapses[j].conduct()
      for j in range(len(self.neurons)) :              #update potentials
        self.vSim[j,i] = self.getV(self.neurons[j], t, self.synapses[j].Gs_1,
                         self.synapses[j].Gs_23,self.synapses[j].Gs_4)
        if self.neurons[j].v >= self.neurons[j].v_th : #record IF spikes
          self.neurons[j].sTime = t
          self.neurons[j].sTrain += [self.ht + t]
          self.raster[j][i] = j
      for p in range(len(self.poissons)):              #check for P-spikes
        if t in self.poissons[p].sTrain :
          self.poissons[p].sTime = t
          self.raster[self.poissons[p].id][i] = self.poissons[p].id
    for j in range(len(self.allNrns)) :                #reset sTimes
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
    self.T      = T

    time = np.arange(0, T, dt)
    for i, t in enumerate(time) :
      x = np.random.rand()
      if x < sRate*dt :
        self.sTrain += [t]
        self.count += 1

  def delay(self, d):
    i = 0
    if d >= self.T or d <= - self.T :
      self.sTrain = []
      return
    while i < len(self.sTrain):
      self.sTrain[i] += d
      if self.sTrain[i] > self.T or self.sTrain[i] < 0:
        self.sTrain.pop(i)
      i += 1


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
# Function implementing Victor-Purpura spike-time metric
# ------------------------------------------------------------------------------
def VP_computeDistance(t1, t2, q) :
# manage corner cases
  if len(t1) == 0 and len(t2) == 0:
    return 1000
  if len(t1) == 0 :
    return len(t2)
  if len(t2) == 0 :
    return len(t1)
# compute using dynamic programming algorithm
  G = np.zeros([len(t1), len(t2)])
  G[:, 0] = [i for i in range(len(t1))]
  G[0, :] = [i for i in range(len(t2))]
  for i in range(1, len(t1)) :
    for j in range(1, len(t2)) :
      G[i][j] = min(G[i-1][j-1] + q*abs(t1[i]-t2[j]), G[i-1][j]+1, G[i][j-1]+1)
  return G[-1][-1]


# ------------------------------------------------------------------------------
# Function implementing van Rossum metric
# ------------------------------------------------------------------------------
def vR_computeDistance(u, v, tau) :
  dist = 0
  for i in range(len(u)):
    for j in range(len(u)):
      dist +=   math.exp(- abs(u[i] - u[j])/tau)
  for i in range(len(v)):
    for j in range(len(v)):
      dist +=   math.exp(- abs(v[i] - v[j])/tau)
  for i in range(len(u)):
    for j in range(len(v)):
      dist -= 2*math.exp(- abs(u[i] - v[j])/tau)
  dist = math.sqrt(dist)
  return dist


# ------------------------------------------------------------------------------
# Filter spike train through exponential kernel (vR form FOR TESTING)
# ------------------------------------------------------------------------------
def vR_mapTrain(spikes, dt, T, tau, mu) :
  time = np.arange(0, T, dt)
  h = np.zeros(len(time))

  for i, t_i in enumerate(time) :
    for j in range(len(spikes)) :
      t = t_i - spikes[j]
      h[i] += (1-mu)*math.exp(-t/tau) if t >= 0 else 0

  return h

# calculate distance inefficiently between spike trains in 'vR form'
def vR_computeDistance_mappedTrains(f1, f2, dt) :
  diff = np.zeros(len(f1))
  dist = 0
  for i in range(len(f1)) :
    diff[i] = f1[i] - f2[i]
    dist += diff[i]*diff[i]*dt

  dist = math.sqrt(dist)

  return dist, diff

# ------------------------------------------------------------------------------
# objects to store the settings for a metric
# ------------------------------------------------------------------------------
class metric(object):
  def __init__(self, name, tau=None, q=None):
    self.type = name

    if name == 'vR':
      self.tau = tau

    elif name == 'VP':
      self.q  = q


#-------------------------------------------------------------------------------
# Compute distance matrix given a set of spike trains and a metric
#-------------------------------------------------------------------------------
def getDistanceMap(trains, metric):
  N = len(trains)
  dMap = np.zeros([N,N])

  for i in range(N):
    for j in range(N):
      if dMap[j][i] != 0 :
        dMap[i][j] = dMap[j][i]
      elif i != j :
        if metric.type == 'vR':
          dMap[i][j] = vR_computeDistance(trains[i], trains[j], metric.tau)
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
  if len(S) != len(R) :
    raise Exception("|S| != |R| !")
  N = len(S)
  if h1 >= N or h2 >= N:
    raise Exception("h1 or h2 >= N !")
  h1h2 = h1*h2
  MI = 0
  dMap_S = getDistanceMap(S, metric)
  dMap_R = getDistanceMap(R, metric)
  for i in range(N):
    b_si = dMap_S[i].argsort()[:h1+1]
    b_ri = dMap_R[i].argsort()[:h2+1]
    count = len(np.intersect1d(b_si, b_ri))
    MI += math.log((N*count)/h1h2, 2)
  MI = MI/N
  return MI
