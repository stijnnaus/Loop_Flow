# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:20:53 2016

@author: naus010
"""

# Testcase obs_oper and adj_obs_oper
test1 = [[1,2,3],[4,5,6],[7,8,9]]
loc_test1 = [0,2]
nx = 3
aa = obs_oper(test1,loc_test1)
bb = adj_obs_oper(aa,loc_test1,nx)
cc = adj_obs_oper([1,2,3],[0,5,9],10)

# Adjoint model

# Fixed parameters
(nx,dx) = (int(1e2), 2e4) # nx grid cells; grid size (m)
ntMax = 30000 # Number of timesteps
dt = 8e1 # nt time steps; time step (s)
kap = 2e6 # Diffusivity coefficient (m2/s)
lamb = 4e-5 # Decay constant 
u = 10 # Advection wind speed (m/s)
cond1 = kap * dt / dx**2
cond2 = u * dx / kap
print cond1, cond2

# Emission parameters
nsource = 4 # Number of sources
loc_source = [ random.randint(0,nx-1) for i in range(nsource) ]  # Source locations (random)
Ewidth = 20 # Peak width emissions
maxStrength = 1e-3 # Max source strength

# Inverse model parameters

# Emission profile
E = EmissionProfile(nx,dt,loc_source,Ewidth,maxStrength)    

# Starting concentration profile (Just 0 everywhere)
Cstart = np.empty(nx)
for i in range(nx):
    Cstart[i] = 0
    
# Calculating the exact model data, using the forward model (for large nt)
C_real = ForwardModel(nx,dx ,ntMax,dt, kap, lamb, u, E, Cstart)

C_true = ForwardModel(nx,dx, nt,dt, kap, lamb, u, x0[nx:], x0[:nx])