from __future__ import division
import numpy as np
import scipy as scp
import matplotlib as mp
from pylab import *
import matplotlib.pyplot as plt

# ==============================================================================
# OBJECT DEFINITIONS
# ==============================================================================

# Defines a neuron object and its voltage membrane function
# ------------------------------------------------------------------------------
class neuron (object):    
  def __init__(self, E_l, V_th, V_res, V_0, R_m, I_e, t_m, t_ref, s_time):
    self.el = E_l          #Reverse potential - l for leak current
    self.vt = V_th         #Spiking Threshold
    self.vr = V_res        #Reset voltage value
    self.rm = R_m          #Resistance of the membrane
    self.i  = I_e          #Constant Input Current
    self.tm = t_m          #Time constant of the membrane Cm/Gm
    self.tr = t_ref        #Refractory period after a spike
    
    self.v = V_0           #Current voltage of the membrane
    
    self.spike_t = s_time  #Time of last spike <- constant
    
    self.synIds = []       #List of indices of synapting neurons  
    self.syns = []         #List of synapse synapting neuron objects
    self.Gs  = 0           #Summed synaptic conductances, default: 0
    self.es   = 0*mili     #Reverse synapse potential, exitory: 0
    
# Return value of f(V) = dV/dt
  def f(self, V):
    #df/dt = (R_m*I - V + E_l - R_m*G_s*(V - E_s) )*G_m/C_m
    return (self.el - V + self.rm*self.Gs*(self.es - V) + self.rm*self.i)/self.tm
    
# Ouptut the membrane voltage at a single timeslice using RK4 appximation
  def getV(self, dt, t):
    
    if self.v >= self.vt :  #and t > self.spike_t + self.tr 
      self.v = self.vr
      self.spike_t = t - dt
      
    k1 = self.f(self.v)
    #k2 = self.f(self.v + k1*dt/2)
    #k3 = self.f(self.v + k2*dt/2)
    #k4 = self.f(self.v + k3*dt)
     
    #self.v += 1/6*(k1 + 2*k2 + 2*k3 + k4)*dt
    self.v += k1*dt
    
    return self.v

      
      
  
# Computes the post-synaptic conductivity incurred by single pre-synaptic neuron
#-------------------------------------------------------------------------------
class synapse:
  def __init__ (self, t, gs=None, t_s=None) :
    self.t  = t            #Time since pre-synaptic neuron's last spike
    
    if gs is None :
      self.gs = 0.15       #Stregth of synapse -> default constant
    else :
      self.gs = gs
   
    if t_s is None :     
      self.ts = 10*mili   #Time scale -> default 10[ms]
    else :
      self.ts = t_s
      
  def conduct(self) :
    return self.gs*exp(-self.t/self.ts)


    

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
    ns = self.ns
    mat = self.mat
    dt - self.dt
    Vsim = self.Vsim
    
    Vsim[:,0] = [o.v for o in ns] #get the initial voltages of neurons        
    
#   Initialise
#   iterate j through neuron ids
    for j in range(len(ns)) :
#     get pre-synaptic neurons ids
      ns[j].synIds = [item for sublist in np.nonzero(mat[j]) for item in sublist] 
#     iterate pre-synaptic neurons ids         
      for n in (ns[j].synIds) :
#       create and append a synapse object for each synapting neuron             
        ns[j].syns.append(synapse(0 - ns[n].spike_t)) 
#       add its current conductance to the sum
        ns[j].Gs += ns[j].syns[-1].conduct()
        
#   Simulate
#   iterate i through time-array indices 
#   iterate t through time steps  
    for i, t in enumerate(self.t[1:], 1) :
#     iterate j through neuron ids       
      for j in range(len(ns)) :
#       update membrane potential of neuron j at time-id i  
        Vsim[j,i] = ns[j].getV(dt, t)
        
#       check for spikes and record spike-timing       
        if ns[j].v > ns[j].vt :
          ns[j].spike_t = t

#     iterate j through neuron ids again after update cycle
      for j in range(len(ns)) :
#       reset synaptic conductivity
        ns[j].Gs = 0
#       iterate si through indeces of neuron j's lists of pre-synaptic neurons
#       iterte ni through ids of pre-synaptic neurons
        for si, ni in enumerate(ns[j].synIds) :
#         update time for each synapse to 
          ns[j].syns[si].t = t - ns[ni].spike_t
#         add corresponding synapse current conductance to sum
          ns[j].Gs += ns[j].syns[si].conduct()
              
    return Vsim




# Simulates a single neuron using the LIF model 
# ------------------------------------------------------------------------------
class neuronSim : 
  def __init__(self, n, T, dt):
    self.neuron = n
    self.dt = dt
    self.t = np.arange(0, T, dt)
    self.sim = zeros(len(self.t))   #membrane voltage at time t
    self.train = zeros(len(self.t)) #spike train based on the simulation
    self.fire_rate = 0
  
# Simulate
  def integrate(self, n=None):
    if n is None : n = self.neuron  #checks if operands point to same object
    sim = self.sim
    train = self.train
    dt = self.dt
    sim[0] = n.v
    train[0] = n.vr-30
      
    for i, t in enumerate(self.t[1:], 1):
      sim[i] = n.getV(dt,t)
      if sim[i] > n.vt :
        self.fire_rate +=1
        train[i] = n.vr-20
      else:
        train[i] = n.vr-30
    return sim




# ==============================================================================
# MAIN SCRIPT       
# ==============================================================================   
    
# Global parameters ------------------------------------------------------------
mili = 0.001          # Scaling factor 10^-3

E_l   = -70*mili      # Standard reverse potential 
V_th  = -40*mili      # Vth = -40 [mV]
V_res = E_l           # Reset membrane potential
R_m   = 1             # Rm = 1[M_ohm] 
I_e   = 31*mili       # Ie = 3.1[nA]
t_m   = 10*mili       # tau_m = 10[ms] = C_m*Rm time constant of the membrane
t_ref = 100*mili        # Refractory period = 5[ms]
dt    = 1*mili        # Time scale: 1 slice = 1[ms]
T     = 1             # simulation period = 1[s]

# Simulate single neuron with constant input current I_e -----------------------
n1 = neuron(E_l, V_th, V_res, V_res, R_m, I_e, t_m, t_ref, -t_ref)
sim = neuronSim(n1, T, dt)
v = sim.integrate()
time = np.arange(0, T, dt)

#print sim.fire_rate:


#plot figure
figure(1)
plot(time, v)
show()

# Simulate a Network -----------------------------------------------------------
I_e   = 28*mili         # I_e = 1.8[nA]
t_m   = 30*mili
V_th  = -50*mili
V_res = -80*mili

#synapse connectivity weight matrix := equally weighted ring
N = 2
synMat = np.eye(N)              #create identity matrix
synMat = np.roll(synMat, -1, 1) #shift one position to create ring

#list to store neurons
neurons = []

#generate random initial membrane voltages and last-spike times
Vs = V_res + np.random.rand(N)*(V_th - V_res)
sTs = -t_ref + np.random.rand(N)*(0, t_ref)

#iterate i through neuron ids
for i in range(N):
# create neuron objects
  neurons += [neuron(E_l, V_th, V_res, Vs[i], R_m, I_e, t_m, t_ref, sTs[i])]

#create network simulaton
ns1 = netSim(neurons, synMat, T, dt)
#simulate network and store results
Vs = ns1.simulate()

#plot figure
figure(2)
p1, = plot(time, Vs[0], 'g')  
p2, = plot(time, Vs[1], 'r')
show()


