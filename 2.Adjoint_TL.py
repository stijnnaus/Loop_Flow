# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:10:35 2016

@author: naus010
"""

import numpy as np
import matplotlib.pyplot as plt
import math as M
from scipy.stats import norm
import random
import scipy.sparse as scs
import time
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc

ntStandard = 1000

def ForwardModelTL(x,nt = ntStandard):
    '''
    The main routine for the forward model, that calculates the concentration 
    profile through time from the emission profile E. The state includes emis-
    sions, concentrations and the wind speed (constant through space and time)
    '''
    E,Cstart,u = x[:nx],x[nx:-1],x[-1]
    
    # Useful constants
    q = kap * dt/dx**2
    q2 = 2*q
    s = 0.5 * u * dt/dx
    lt = lamb*dt
    
    C = np.empty((nt+1,nx)) # Concentrations
    C[0] = Cstart
    for n in range(nt):
        for m in range(nx):
            C[n+1][m] = E[m] + (1 - lt - q2) * C[n][m] + ( q - s ) * C[n][(m+1)%nx] + ( q + s ) * C[n][m-1]
            
    return np.array(C[1:])

def AdjointModelTL(mismatch, Csaved, usaved):
    '''
    Uses the adjoint tangent linear model to calculate the second term in dJ/dx.
    Returns HT * R^(-1) * (Hx - y) (2nx x 1: one part of dJ/dx)
    ''' 
    
    u = usaved
    
    # Useful constants 
    nobs = len(loc_obsA)
    a = kap * dt/dx**2 + 0.5*u*dt/dx
    b = 1 - lamb*dt - 2 * kap * dt/dx**2
    c = kap * dt/dx**2 - 0.5*u*dt/dx
    s = 0.5*dt/dx
    
    # Sensitivities
    dC = np.zeros(nx)
    dE = np.zeros(nx)
    dU = 0.
    
    pulses = mismatch/Oerror**2
    
    # Run the adjoint model
    for i in range(ntA-1,0,-1):
        Ci = Csaved[i]
        
        # Add the adjoint pulse:
        dC += adj_obs_oper( pulses[i] )
        
        # Emission step
        dE += dC
        
        # Advection wind
        for m in range(nx):
            dU += s * ( Ci[m-1] - Ci[(m+1)%nx] ) * dC[m]
            
        # Advection/diffusion/radioactive decay step
        dC_temp = np.empty(nx)
        for m in range(nx):
            dC_temp[m] = c * dC[m-1] + b * dC[m] + a * dC[(m+1) % nx]
        dC = dC_temp.copy()
    print 'E:',np.max(dE),'C:',np.max(dC),'U:',np.max(dU)
    return tostate(dE,dC,dU)
    
    
def obs_oper(C):
    '''
    Maps a field C to the observation stations.
    Can work with a 2D (eg nt x nx) array, or with 1D (eg nx).
    Returns an nobs x nt array, with each row corresponding to an observation
    station.
    '''
    C = np.array(C)
    Cobs = []
    Cshape = len(C.shape)
    # 2D array
    if Cshape == 2:
        nt = len(C)
        nx = len(C[0])
        for i in range(nt):
            Cobsi = []
            for loc in loc_obsA:
                Cobsi.append(C[i][loc])
            Cobs.append(Cobsi)
            
    
    # 1D array
    elif Cshape == 1:
        nx = len(C)
        for loc in loc_obsA:
            Cobs.append([C[loc]])
    
    else:
        print 'Invalid array shape.'
        return 
    
    return np.array(Cobs)

def adj_obs_oper(C_obs):
    '''
    Maps observations (nobs x nt) to the nx grid.
    Works for 1D (1 timestep) and 2D (nt timesteps) arrays.
    Returns nt x nx array.
    '''
    C_obs = np.array(C_obs)
    Oshape = len(C_obs.shape)
    if Oshape == 1: # 1D array (1 timestep)
        Cout = np.zeros(nx)
        for m,loc in enumerate(loc_obsA):
            Cout[loc] = C_obs[m]
    
    elif Oshape == 2: # 2D array (nt timesteps)
        nt = len(C_obs[0])
        Cout = np.zeros((nt,nx))
        for i in range(nt):
            for m,loc in enumerate(loc_obsA):
                Cout[i][loc] = C_obs[m][i]            
    return Cout
    
def calc_mismatchTL(x):
    '''
    Calculate the mismatch between C_obs calculated from E0 and C0 forward
    in time and y (observation data).
    Returns the mismatch and the forward concentrations. 
    '''
    
    C = ForwardModelTL(x, nt = ntA)
    C_obs = obs_oper(C)
    mismatch = np.array(C_obs - dataA)
    return mismatch,C

reduction = 1e-8
J_obs = []
J_prior = []
def calc_JTL(x0):
    '''
    Calculates the cost function J from the guess x0.
    '''
    mismatch,local = calc_mismatchTL(x0)
    delx = x0 - x_priorA
    Jobs = 0.5*np.sum((mismatch / Oerror) **2)
    Jprior = 0.5 * np.dot(delx, np.dot(B_inv, delx))
    J_obs.append(Jobs)
    J_prior.append(J_prior)
    print 'Jobs, Jprior = ',Jobs,Jprior
    #print 'x0: ', np.sum(abs(x0))
    print 'J:',(Jobs + Jprior)*reduction
    return (Jobs + Jprior)*reduction

def calc_JTL2(x0):
    '''
    Calculates the cost function J from the guess x0.
    '''
    mismatch,local = calc_mismatchTL(x0)
    delx = x0 - x_priorA
    Jobs = 0.5*np.sum((mismatch / Oerror) **2)
    Jprior = 0.5 * np.dot(delx, np.dot(B_inv, delx))
    J_obs.append(Jobs)
    J_prior.append(J_prior)
    return delx, np.dot(B_inv, delx)
    

    
def calc_dJdxTL(x0):
    '''
    Computes derivative dJ/dx from the guess x0
    '''
    mismatch,Csaved = calc_mismatchTL(x0)
    u_old = x0[-1]
    adjoint = AdjointModelTL( mismatch , Csaved, u_old )
    delx = x0 - x_priorA
    dJdx = np.dot(B_inv, delx) + adjoint 
    #print 'dJdx (adjoint, prior) =',np.sum(adjoint),np.sum(np.dot(B_inv,delx))
    print 'deriv:',max(abs(dJdx))*reduction
    print 'uderiv:',dJdx[-1]*reduction
    return dJdx*reduction
    
def tostate(E,C,u):
    xa = np.concatenate((E,C))
    x = np.append(xa,u)
    return x

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initialization
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
times = []

experiment = 'TigherE'

# ------------------------------------------------------------------------------
# Parameters that are fixed through each individual run
# Simulation parameters
precon = False              # Preconditioning on or off
(nx,dx) = (int(1e2), 2e4)   # nx grid cells; grid size (m)
ntMax = 30000               # Maximum number of timesteps
ntSteady = 3000             # Number of steps to steady state
dt = 8e1                    # time step (s)
kap = 2e6                   # Diffusivity coefficient (m2/s)
lamb = 4e-5                 # Decay constant 
utrue = 10                  # True advection wind speed (m/s)
cond1 = kap * dt / dx**2    # Stability criterion 1
cond2 = utrue * dx / kap        # Stability criterion 2
print cond1, cond2
# Emission parameters
nsource = 4                 # Number of sources
loc_source = [ random.randint(0,nx-1) for i in range(nsource) ]  # Source locations (random)
Ewidth = 3                 # Peak width for each (Gaussian) source
maxStrength = 1e-3          # Max source strength
# Inverse model parameters
E_true = EmissionProfile(loc_source,Ewidth,maxStrength) # True emission profile
Cstart = np.zeros(nx) # Starting concentration profile (Just 0 everywhere)  
xtrue = tostate(E_true,Cstart,utrue)
C_real = ForwardModelTL(xtrue, nt = ntMax) # The exact model data
C_steady = ForwardModelTL(xtrue, nt = ntSteady) # Steady state model data
# Errors
Eerror = 1e-4 * max(E_true)          # emission error
Cerror = 5e-8 # prior error
Oerror = 1e-3 # observation error as used in the inversions
Uerror = 10.
Oerror_real = Oerror # the actual imposed error
# Construct prior error (/variance) matrix B, possibly with preconditioning
priorError = np.array([Eerror**2]*nx+[Cerror**2]*nx+[Uerror**2])
Bmatrix = np.matrix( np.diag(priorError) )
Barray = np.array( Bmatrix )
Bmatrix_inv = np.linalg.inv(Bmatrix)
B_inv = np.array(Bmatrix_inv)


nobsStandard = 3 # Number of measurement stations
loc_obsStandard = genStations(nobsStandard, rand = False) # Standard locations meas stations
# Generating input data
data = genData(C_real, loc_obsStandard) # Observations 
x0 = np.random.multivariate_normal(xtrue,Bmatrix)
x0[-1] = 0.
C_priorForward = ForwardModelTL(x0,nt = ntMax) # Perturbed model data

plt.plot(x0[:-1])
plt.plot(xtrue[:-1])


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Adjoint TL model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print 'Running the Adjoint Tangent Linear model......'
# ---------------------------------------------------------------
# STANDARD RUN

tol = 1e-5
loc_obsA = loc_obsStandard
nobsA = len(loc_obsA)
ntA = ntStandard
x_priorA = x0.copy()
dataA = np.array( np.split( data[:nobsA*ntA], ntA ) )
x_opt = sc.optimize.fmin_cg(calc_JTL, x0, calc_dJdxTL,gtol = 1e-3, maxiter = 80)
E_resA = x_opt[:nx]
C_resA = x_opt[nx:-1]
U_resA = x_opt[-1]

u = U_resA
C_forwA = ForwardModelTL(x_opt, ntA)

plt.figure()
#for i,x in enumerate(x_opt[1]):
#    if i%3 == 0:
#        plt.plot(x[:-1],label = 'fit' + str(i))
plt.plot(x_opt[:-1], label = 'Final')
plt.plot(x_priorA[:-1], label = 'Prior')
plt.plot(xtrue[:-1])
plt.legend(loc = 'best')

x = [i*dx/1000. for i in range(nx)]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(x, C_priorForward[ntStandard-1], label = 'Prior')
ax1.plot(x, C_forwA[ntStandard-1], label = 'Adjoint')
ax1.plot(x, C_real[ntStandard-1], label = 'Exact')
for i,loc in enumerate(loc_obsA):
    ax1.plot(loc*dx/1000, dataA[-1][i],'ro',markersize = 7)
ax1.set_title("Concentrations Adjoint Tangent Linear")
ax1.set_xlabel("Position (in km)")
ax1.set_ylabel("Concentrations (unitless)")
ax1.legend(loc='best')
plt.savefig('AdjointTL_Concentrations_TightE')

fig_emi = plt.figure()
ax1 = fig_emi.add_subplot(111)
ax1.plot(x, E_resA,'b', label = 'Adjoint')
ax1.plot(x, x0[:nx],'r' ,label = 'Prior')
ax1.plot(x, E_true,'g', label = 'Exact')
ax1.plot(loc_obsStandard*dx/1000.,[0]*len(loc_obsStandard),'bo',markersize=14,label = 'Stations')
ax1.set_title("Resulting emissions. nt = 500, nobs = 3")
ax1.set_xlabel("Position (in km)")
ax1.set_ylabel("Emissions (unitless)")
ax1.legend(loc='best')
plt.savefig('AdjointTL_Emissions_'+experiment)

print 'Prior u: ',x0[-1]
print 'Posterior u: ',x_opt[-1]