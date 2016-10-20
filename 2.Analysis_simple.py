# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:57:52 2016

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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initialization
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
seed = 1
times = []
randomrun = False # If true, a run with randomly distributed stations is also done
random.seed(seed)

# Which methods to run
run_BLUE = True
run_KF = True
run_KE = True
precon = True # Preconditioning on or off

# ------------------------------------------------------------------------------
# Parameters that are fixed through each individual run
# Simulation parameters
(nx,dx) = (int(1e2), 2e4)   # nx grid cells; grid size (m)
ntMax = 3000                # Maximum number of timesteps
ntSteady = 3000             # Number of steps to steady state
dt = 8e1                    # time step (s)
kap = 2e6                   # Diffusivity coefficient (m2/s)
lamb = 4e-5                 # Decay constant 
u = 10                      # Advection wind speed (m/s)
cond1 = kap * dt / dx**2    # Stability criterion 1
cond2 = u * dx / kap        # Stability criterion 2
print cond1, cond2
# Emission parameters
nsource = 4                 # Number of sources
loc_source = [ random.randint(0,nx-1) for i in range(nsource) ]  # Source locations (random)
Ewidth = 3                  # Peak width for each (Gaussian) source
maxStrength = 1e-3          # Max source strength
# Inverse model parameters
E_true = EmissionProfile(loc_source,Ewidth,maxStrength) # True emission profile
Cstart = np.zeros(nx) # Starting concentration profile (Just 0 everywhere)  
xtrue = np.concatenate((E_true,Cstart))
C_real = ForwardModel(E_true, Cstart, nt = ntMax) # The exact model data
C_steady = ForwardModel(E_true, Cstart, nt = ntSteady) # Steady state model data
# Errors
Eerror = 0.1 * max(E_true) # emission error
Cerror = 5e-8 # prior error
Oerror = 1e-3 # observation error as used in the inversions
Oerror_real = Oerror # the actual imposed error
# Construct prior error (/variance) matrix B, possibly with preconditioning
priorError = np.array([Eerror**2]*nx+[Cerror**2]*nx)
Bmatrix = np.matrix( np.diag(priorError) )
if precon: # Preconditioning
    cor_lenE = .01
    cor_lenC = .01
    corrE = [np.exp(-((i/cor_lenE))) for i in range(1,nx+1)]
    corrC = [np.exp(-((i/cor_lenC))) for i in range(1,nx+1)]
    Bmatrix = preCon( Bmatrix,corrE,corrC )
Bmatrix_inv = np.linalg.inv(Bmatrix)
B_inv = np.array(Bmatrix_inv)

nobsStandard = 3 # Number of measurement stations
loc_obsStandard = genStations(nobsStandard, rand = False) # Standard locations meas stations
# Generating input data
dataStandard = genData(C_real[:ntStandard], loc_obsStandard) # Observations 
x0 = np.random.multivariate_normal(xtrue,Bmatrix)
E_prior,C0_prior = x0[:nx],x0[nx:]
C_priorForward = ForwardModel(E_prior,C0_prior,nt = ntMax) # Perturbed model data

plt.plot(x0)
plt.plot(xtrue)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# BLUE calculations
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if run_BLUE:
    print 'Running BLUE....'
    
    E_resB,C_resB,B_B = BLUEAnalysis(dataStandard, loc_obsStandard,nt=ntStandard)
    C_forwB = ForwardModel(E_resB, C_resB ,nt=ntStandard)



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Kalman inversion calculations
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if run_KF:
    print 'Running the Kalman filter .....' 
    
    E_resKF,C_resKF,B_KF = Kalman(dataStandard, loc_obsStandard,nt=ntStandard)
    C_forwKF = ForwardModel(E_resKF, [0]*nx ,nt=ntStandard)
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Kalman ensemble
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if run_KE:
    print 'Running the Kalman ensemble......'

    NN = 2000
    
    E_resKE,C_resKE = EnsembleKalman(dataStandard, loc_obsStandard,NN,nt=ntStandard)
    C_forwKE = ForwardModel(E_resKE, [0]*nx ,nt=ntStandard)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Adjoint model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print 'Running the Adjoint model......'
# ---------------------------------------------------------------
# STANDARD RUN

tol = 1e-4
loc_obsA = loc_obsStandard
nobsA = len(loc_obsA)
ntA = ntStandard
x_priorA = x0
dataA = np.array( np.split(dataStandard,ntA) )

L = np.sqrt( np.array( Bmatrix ) ) # Preconditioning matrix L
L_inv, L_adj = np.linalg.inv(L), np.transpose(L)
xp = state_to_precon(x0) # Preconditioned prior state
xp_opt = sc.optimize.fmin_cg( calc_J, xp, calc_dJdx, gtol = tol , maxiter = 80)
x_opt  = precon_to_state(xp_opt)

E_resA = x_opt[:nx]
C_resA = x_opt[nx:]

C_forwA = ForwardModel(E_resA, C_resA, ntA)

# PLOTS

# CONCENTRATIONS
x = [i*dx/1000. for i in range(nx)]
fig_con = plt.figure()
ax1 = fig_con.add_subplot(111)
ax1.plot(x, C_forwB[-1],'b--',label='BLUE')
ax1.plot(x, C_forwKE[-1],'c--',label = 'Kalman filter')
ax1.plot(x, C_forwKF[-1],'m--',label = 'Kalman Ensemble')
ax1.plot(x, C_forwA[-1],'k--', label = 'Adjoint')
ax1.plot(x, C_priorForward[ntStandard-1],'r', label = 'Prior')
ax1.plot(x, C_real[ntStandard-1],'g', label = 'Exact')
for i,loc in enumerate(loc_obsA):
    ax1.plot(loc*dx/1000, dataA[-1][i],'ro',markersize = 7)
ax1.plot(loc_obsStandard*dx/1000.,[0]*len(loc_obsStandard),'bo',markersize=14,label = 'Stations')
plt.legend(loc = 'best')
ax1.set_title("Concentrations from different approaches (nt = 500, nobs = 3)")
ax1.set_xlabel("Position (in km)")
ax1.set_ylabel("Concentrations (unitless)")
ax1.legend(loc='best')
plt.savefig( 'Simple_Concentrations')

# EMISSIONS
fig_emi = plt.figure()
ax1 = fig_emi.add_subplot(111)
ax1.plot(x, E_resB,label='BLUE')
ax1.plot(x, E_prior, label = 'Prior')
#ax1.plot(x, E_resKF,label = 'Kalman Filter')
ax1.plot(x, E_resKE,label = 'Kalman Ensemble')
ax1.plot(x, E_resA, label = 'Adjoint')
ax1.plot(x, E_true, label = 'Exact')
ax1.plot(loc_obsStandard*dx/1000.,[0]*len(loc_obsStandard),'bo',markersize=14,label = 'Stations')
ax1.set_title("Resulting emissions. nt = 500, nobs = 3")
ax1.set_xlabel("Position (in km)")
ax1.set_ylabel("Emissions (unitless)")
ax1.legend(loc='best')
plt.savefig("Simple_Emissions")

mm1,mm2 = resiCompleteSingle(E_resA, C_forwA, E_true,C_real[:ntStandard],E_prior,C_priorForward)

print mm1,mm2






















    
    
    
    
    
    
    
    
    