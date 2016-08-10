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
               i=None, tm=None, tref=None, se=None, si=None):
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
    self.e_s    = 0     if se   == None else se     #exciting synapt potential
    self.i_s    = -0.08 if si   == None else si     #inhibiting synapt potential
    
# returns value of f(V) = dV/dt
  def f(self, V, G_es, G_is):
    f = ( self.e_l - V + self.r_m*(G_es*(self.e_s - V) + G_is*(self.i_s - V)) +
          self.r_m*self.i ) / self.t_m
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
    self.Gs_1  = 0                            #summed synaptic conductance
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
  def __init__(self, cMat, nrns, synsE, synsI):
    self.cMat = cMat
#   iterate through neuron/synapse ids
    for j in range(len(synsE)):
#     iterate through indices of exciting and inhibiting pre-synaptic neurons
#     to neuron j and fill lists of synaptic strengths and last-spike times
      for e in ( self.list_pre_synapt_ns_exci(synsE[j]) ):
        synsE[j].sGs.append(cMat[e][j])
        synsE[j].sTs.append(0 - nrns[e].sTime)

      for i in ( self.list_pre_synapt_ns_inhi(synsI[j]) ):
        synsI[j].sGs.append(cMat[i][j])
        synsI[j].sTs.append(0 - nrns[i].sTime)

# update times since pre-synaptic spikes in both synapses
  def update_sTs(self, syn_exci, syn_inhi, nrns, t):
      for j,k in enumerate(self.list_pre_synapt_ns_exci(syn_exci)):
        syn_exci.sTs[j] = t - nrns[k].sTime
      for j,k in enumerate(self.list_pre_synapt_ns_inhi(syn_inhi)):
        syn_inhi.sTs[j] = t - nrns[k].sTime
        
# return pre-synaptic neurons reaching parameter synapse
  def list_pre_synapt_ns_exci (self, synapse):
    cMat = self.cMat; id = synapse.id
    return [item for sublist in np.nonzero(cMat[:,id] > 0) for item in sublist]

  def list_pre_synapt_ns_inhi (self, synapse):
    cMat = self.cMat; id = synapse.id
    return [item for sublist in np.nonzero(cMat[:,id] < 0) for item in sublist]

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
    self.synsExci = []
    self.synsInhi = []
    self.cMat     = cMat                     #connectivity matrix
    self.t        = np.arange(0, T, dt)      #time array
    self.dt       = dt                       #time step
    self.ht       = 0 if h_t==None else h_t  #chronologic time incurred
    
    for i in range(len(self.allNrns)) :      #neurons and synapse objects
      if self.allNrns[i].type == 'IF':
        if h_t == None :
           self.allNrns[i].sTrain = []
        self.neurons.append(self.allNrns[i])
        self.synsExci.append(synapse(i, self.dt))
        self.synsInhi.append(synapse(i, self.dt))
      elif Nrns[i].type == 'P':
        self.poissons.append(self.allNrns[i])
        
    self.cNet   = cnet(cMat, self.allNrns, self.synsExci, self.synsInhi)
#   arrays storing simulation data - potential and conductivity over time
    self.vSim   = np.zeros([len(self.neurons), len(self.t)])
    self.gSimE  = np.zeros([len(self.neurons), len(self.t)])
    self.gSimI  = np.zeros([len(self.neurons), len(self.t)])
    self.raster = np.zeros([len(self.allNrns), len(self.t)])*np.nan

# Compute membrane potential at a single timeslice using RK4 appximation
  def getV(self, nrn, t, Gs_1e, Gs_1i, Gs_23e, Gs_23i, Gs_4e, Gs_4i):
    if nrn.v >= nrn.v_th :             #track spikes and reset potential
      nrn.v = nrn.v_res
    elif t < nrn.sTime + nrn.t_ref :   #hold reset if refracory period
      nrn.v = nrn.v_res
    else :                             #inegreate using RK4 method
      k1 = self.dt*nrn.f(nrn.v, Gs_1e, Gs_1i )
      k2 = self.dt*nrn.f(nrn.v + k1/2, Gs_23e, Gs_23i)
      k3 = self.dt*nrn.f(nrn.v + k2/2, Gs_23e, Gs_23i)
      k4 = self.dt*nrn.f(nrn.v + k3, Gs_4e, Gs_4i)
      nrn.v += 1/6*(k1 + 2*k2 + 2*k3 + k4)
    return nrn.v

# Simulate network
  def simulate(self):
    for i, t in enumerate(self.t) :
      for j in range(len(self.neurons)) :             #update conductances
        self.cNet.update_sTs(self.synsExci[j], self.synsInhi[j], self.allNrns,t)
        self.gSimE[j,i] = self.synsExci[j].conduct()
        self.gSimI[j,i] = self.synsInhi[j].conduct()
        
      for j in range(len(self.neurons)) :              #update potentials
        self.vSim[j,i] = self.getV(self.neurons[j], t, 
                                 self.synsExci[j].Gs_1,  self.synsInhi[j].Gs_1,
                                 self.synsExci[j].Gs_23, self.synsInhi[j].Gs_23,
                                 self.synsExci[j].Gs_4,  self.synsInhi[j].Gs_4)
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
    return self.vSim, self.gSimE, self.raster


# ------------------------------------------------------------------------------
# Class generating Poisson neuron
# ------------------------------------------------------------------------------
class pNeuron (object):
  def __init__(self, idx, sRate, T, dt, st, label=None):
    self.id     = idx            #index in connectivity matrix
    self.type   = 'P'
    self.sTime  = st             #time of last spike
    self.T      = T
    self.label  = label

    self.sTrain, self.count = getPoissonTrain2(T, dt, sRate)

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


def getPoissonTrain2(T, dt, sRate) :
  train = []
  scale = 1/sRate
  count = 0
  t = 0

  while t < T :
    interspike = np.random.exponential(scale)
    t += round(interspike, 3)
    if t <= T :
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

  if (dist > 0) :
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


# ------------------------------------------------------------------------------
# Compute MI between stimuli and responses using the Bialek method
# ------------------------------------------------------------------------------
# Map spike-train fragments to binary words (for the Bialek method)
def B_parseTrain(spikes, dt, T, fsize) :
  fno = int(T/fsize)
  if fno != T/fsize :
    raise Exception("Frame length doesn't divide spike train period!")

  flen = int(fsize/dt)
  if flen != fsize/dt :
    raise Exception("Bin size doesn't divide frame length!")

  time = np.arange(0, T, dt)   # spike train time span array
  word = ""                    # spike train as binary word
  s_i = 0                      # spike idx // MAKE GENERIC IN TERMS OF dt

  for i, t_i in enumerate(time) :
    if spikes[s_i] >= t_i and spikes[s_i] < (t_i+dt):
      word += '1'
      s_i += 1
      if s_i == len(spikes) :
       word += '0'*(len(time) - i - 1)
       break
    else :
      word += '0'

#  print word
  words = [int(word[i:i+flen], 2) for i in range(0, len(word), flen)]

  return words


# Compute the total entropy of a set of spike-train segments
def B_computeS(R) :
  N = len(R)*len(R[0])
# use a dictionary mapping distinct segments to their number of ocurrances
  D = {}
  for i in range(len(R)) :
    for f in range(len(R[i])) :
      if R[i][f] in D :
        D[R[i][f]] += 1
      else :
        D[R[i][f]]  = 1
# Compute the entropy of all frames over the set spike train responses
  S = 0
  for w in D :
    p_w = D[w]/N
    S -= p_w*math.log(p_w, 2)

  return S


# Compute the conditional or noise entropy of Responses given Stimuli segments
def B_computeN(S, R, times) :
  D = {}
  stim = S[0].label
  n = times*len(R[0])     # total no. of frames in all responses to a stimulus
  T = int(len(S)/times)   # number of stimuli
  N = 0                   # Conditional / Noise entropy

  for i in range(len(S)) :
    if S[i].label != stim :
      SH = 0
      for w in D :
        p_w  = D[w]/n
        SH -= p_w*math.log(p_w, 2)
      N += SH/T
      stim = S[i].label
      D = {}

    for f in range(len(R[i])) :
      if R[i][f] in D :
        D[R[i][f]] += 1
      else :
        D[R[i][f]]  = 1

  return N


# Compute MI using Bialek as the difference between S and N
def B_MI(S, R, dt, T, fsize, times) :
  if len(S) != len(R) :
     raise Exception("|S| != |R| !")

  rWords=[]
  for i in range(len(R)) :
    rWords.append(B_parseTrain(R[i], dt, T, fsize))
  rWords = np.array(rWords)

  H  = B_computeS(rWords)
  print 'S = ' +str(H)
  CH = B_computeN(S, rWords, times)
  print 'N = ' + str(CH)
  MI  = H - CH
  print 'I = ' + str(MI)
  print '\n'

  return MI


# ------------------------------------------------------------------------------
# Compute mutual information between two sets of spike trains
# given a metric and a resolution for the model
# ------------------------------------------------------------------------------
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
