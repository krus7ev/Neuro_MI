from __future__ import division
import numpy as np
import scipy as scp
import matplotlib as mp
from pylab import *
import matplotlib.pyplot as plt

# LIF neuron - parameters, state, functions computing membrane potential  
# ------------------------------------------------------------------------------
class neuron (object):    
  def __init__(self, E_l, V_th, V_res, V_0, R_m, I_e, t_m, t_ref, s_time, 
               E_s=0, slow_K=0, E_K=-80*0.001, t_K = 200*0.001):
    self.el = E_l          
    self.vt = V_th         
    self.vr = V_res        
    self.rm = R_m          
    self.i  = I_e          
    self.tm = t_m          
    self.tr = t_ref        
    self.v  = V_0           #current voltage of the membrane
      
    self.spike_t = s_time   #time of last spike <- constant
    self.synIds  = []       #list of indices of neurons synapting to this neuron
    self.syns    = []       #list of synapses (objects) to this neuron
    self.Gs      = 0        #summed synaptic conductance
    self.es      = E_s      #reverse synapse potential (exitatory / inhibitory)
    
    self.sK = slow_K        #slow potassium channel boolean  
    self.ek = E_K
    self.tk = t_K 
    self.Gk = 0
    
  def f(self, V):
    if self.sK : 
      return (self.el - V + self.rm*self.Gs*(self.es - V) + 
              self.rm*self.Gk*(self.ek - V) + self.rm*self.i)/self.tm
    else:
      return (self.el - V + self.rm*self.Gs*(self.es - V) + 
              self.rm*self.i)/self.tm
    
# Compute the membrane potential at a given time using the Euler method
  def getV(self, dt, t):
    if self.v > self.vt and t > self.spike_t + self.tr :
      self.v = self.vr
      self.spike_t = t - dt
      if self.sK:
        self.Gk += 0.005/Mega
      
    self.v += self.f(self.v)*dt
    if self.sK:
      self.Gk += -self.Gk/self.tk*dt
    return self.v

      
# Synaptic input - computes conductance incurred by a pre-synaptic neuron
#-------------------------------------------------------------------------------
class synapse:
  def __init__ (self, t, gs=None, t_s=None) :
    self.t  = t                  #time since pre-synaptic neuron's last spike   
    if gs is None :
      self.gs = 0.015/Mega       #default synapse strength
    else :
      self.gs = gs
      
    if t_s is None :     
      self.ts = 10*mili          #default time-scale
    else :
      self.ts = t_s
      
  def conduct(self) :
    return 0.5*self.gs*exp(-self.t/self.ts)


# Simulates a network of neurons given as a list and a connectivity matrix
# ------------------------------------------------------------------------------     
class netSim :
  def __init__(self, ns, mat, T, dt):     
    self.ns  = ns
    self.mat = mat
    self.t   = np.arange(0, T, dt)
    self.dt  = dt 
    self.Vsim = np.zeros([len(ns), len(self.t)])
   
# Simulate network  
  def simulate(self):
#   Initialise voltages and conductances      
    self.Vsim[:,0] = [n.v for n in self.ns]   
    for j in range(len(self.ns)) :
      self.ns[j].synIds = [item for sublist in np.nonzero(self.mat[j])
                                for item in sublist] 
#     iterate pre-synaptic neurons ids         
      for n in (self.ns[j].synIds) :
#       create and append a synapse object for each synapting neuron             
        self.ns[j].syns.append(synapse(0 - self.ns[n].spike_t)) 
#       add its current conductance to the sum
        self.ns[j].Gs += self.ns[j].syns[-1].conduct()
        
#   Simulate 
    for i, t in enumerate(self.t[1:], 1) :       
      for j in range(len(self.ns)) :
#       update membrane potential of neuron j at time-id i  
        self.Vsim[j,i] = self.ns[j].getV(dt, t)
        
#       check and update last-spike timestamp       
        if self.ns[j].v > self.ns[j].vt :
          self.ns[j].spike_t = t

#     iterate neuron ids again
      for j in range(len(self.ns)) :
#       reset total synaptic conductivity
        self.ns[j].Gs = 0
#       iterate indices in j's list of pre-synaptic neurons and their ids
        for si, ni in enumerate(self.ns[j].synIds) :
#         update timestamps in synapse and incur conductance
          self.ns[j].syns[si].t = t - self.ns[ni].spike_t
          self.ns[j].Gs += self.ns[j].syns[si].conduct()
              
    return self.Vsim


# Simulates a single LIF neuron 
# ------------------------------------------------------------------------------
class neuronSim : 
  def __init__(self, n, T, dt):
    self.neuron = n
    self.dt = dt
    self.t = np.arange(0, T, dt)
    self.sim = zeros(len(self.t))   #membrane voltage at time t
    self.fire_rate = 0
  
# Simulate
  def integrate(self):
    self.fire_rate = 0
    self.sim[0] = self.neuron.v
      
    for i, t in enumerate(self.t[1:], 1):
      self.sim[i] = self.neuron.getV(dt,t)
      if self.sim[i] > self.neuron.vt :
        self.fire_rate +=1

    return self.sim


# ------------------------------------------------------------------------------     
# Main Procedure       
# ------------------------------------------------------------------------------     
# ----------------------Global parameters --------------------------------------
mili = 0.001          # scaling factor 10^-3
Mega = 10**6          # scaling factor 10^6
nano = 0.000000001    # scaling factor 10^-9

E_l   = -70*mili      # reverse potential [mV]
V_th  = -40*mili      # threshold voltage [mV]
V_res = E_l           # reset membrane potential [mV]
R_m   = 10*Mega       # resistane of te membrane [M_ohm] 
I_e   = 3.1*nano      # constant input current [nA]
t_m   = 10*mili       # time constant of the membrane = C_m*Rm [ms]
t_ref = 0*mili        # refractory period [ms]
dt    = 1*mili        # time scale [ms]
T     = 1             # simulation period [s]

# ------------------Q1: Single LIF neuron simulation----------------------------
n1   = neuron(E_l, V_th, V_res, V_res, R_m, I_e, t_m, t_ref, t_ref)
sim  = neuronSim(n1, T, dt)
v1   = sim.integrate()
time = np.arange(0, T, dt)

figure(1)
plot(time, v1)
show()

# ------------------Q2: Mminimum current for a skpike---------------------------
# 1)Set the potential higher than the threshold
#   E_l + R_m * I_e > V_th
# 2)Rearange to express I_e
#   I_e > (V_th - E_l) / R_m
# 3)Substituting in
#   I_e > (-40 + 70)mV / 10 = 3 mV
#   I_min ~ 3.001 mV

# ------------------Q3: Lower-current stimulation-------------------------------
I_e = 2.901*nano
n2  = neuron(E_l, V_th, V_res, V_res, R_m, I_e, t_m, t_ref, t_ref)
sim = neuronSim(n2, T, dt)
v2  = sim.integrate()

figure(2)
plot (time,v2)
show()

# ------------------Q4: Ranging current simulation------------------------------
# Record and plot fire rates as function of the current
f_rates = []
for i in range (20, 51):
  n4 = neuron(E_l, V_th, V_res, V_res, R_m, I_e, t_m, t_ref, t_ref)
  n4.i = i*0.1*nano
  sim = neuronSim(n4, T, dt)
  v4 = sim.integrate()
  f_rates += [sim.fire_rate]
    
Is = np.linspace(2.0, 5.0, 31)

figure(3)
plot(Is, f_rates)
show()

# ------------------Q5: Coupled Neurons simulation------------------------------
I_e   = 1.8*nano
t_m   = 20*mili
V_th  = -54*mili
V_res = -80*mili
N = 2
synMat = np.eye(N)                  # connectivity matrix := identity
synMat = np.roll(synMat, -1, 1)     # shift by 1 to create ring (coupling)

neurons = []
Vs = V_res + np.random.rand(N)*(V_th - V_res)
sTs = -t_ref + np.random.rand(N)*(0, t_ref)
for i in range(N):
  neurons += [neuron(E_l, V_th, V_res, Vs[i], R_m, I_e, t_m, t_ref, sTs[i])]

ns1 = netSim(neurons, synMat, T, dt)
Vs = ns1.simulate()

figure(4)
p1, = plot(time, Vs[0], 'g')  
p2, = plot(time, Vs[1], 'r')
show()


# ------------------Q6: LIF neuron with slow potassium current------------------
I_e   = 3.1*nano
t_m   = 10*mili
V_th  = -40*mili
V_res = E_l
n6    = neuron(E_l, V_th, V_res, V_res, R_m, I_e, t_m, t_ref, t_ref, slow_K = 1)
sim   = neuronSim(n6, T, dt)
v6    = sim.integrate()

figure(5)
plot(time, v6)
show()
