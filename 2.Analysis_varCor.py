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
Bmatrix_start = np.matrix( np.diag(priorError) )
Bmatrices = []
cor_len = [0.01,1.,3.,8.,12.]
for cor in cor_len:
    corr = [np.exp(-((i/cor))) for i in range(1,nx+1)]
    Bmatrixi = preCon( Bmatrix_start,corr,corr )
    Bmatrices.append(Bmatrixi)
    
nobsStandard = 3 # Number of measurement stations
loc_obsStandard = genStations(nobsStandard, rand = False) # Standard locations meas stations
# Generating input data
dataStandard = genData(C_real[:ntStandard], loc_obsStandard) # Observations 

E_priors,C0_priors,C_priorForwards = [],[],[]
for Bmatrixi in Bmatrices:
    x0 = np.random.multivariate_normal(xtrue,Bmatrixi)
    E_priori,C0_priori = x0[:nx],x0[nx:]
    C_priorForwardi = ForwardModel(E_priori,C0_priori,nt = ntMax) # Perturbed model data
    E_priors.append(E_priori)
    C0_priors.append(C0_priori)
    C_priorForwards.append(C_priorForwardi)

E_resB, E_resKF, E_resKE, E_resA = [],[],[],[] # Optimized emissions
C_resB, C_resKF, C_resKE, C_resA = [],[],[],[] # Optimized starting concentrations
C_forB, C_forKF, C_forKE, C_forA = [],[],[],[] # Forward concentrations
E_residB, E_residKF, E_residKE, E_residA = [],[],[],[] # Residuals
C_residB, C_residKF, C_residKE, C_residA = [],[],[],[] # Residuals
savL = []
for ii,Bmatrixi in enumerate(Bmatrices):
    
    Bmatrix, Bmatrix_inv = Bmatrixi, np.linalg.inv(Bmatrixi)
    B_inv = np.array(Bmatrix_inv)
    E_prior,C0_prior,C_priorForward = E_priors[ii],C0_priors[ii],C_priorForwards[ii]
    x0 = np.concatenate((E_prior,C0_prior))
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # BLUE calculations
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    if run_BLUE:
        print 'Running BLUE....'
        
        E_resBi,C_resBi,B_B = BLUEAnalysis(dataStandard, loc_obsStandard,nt=ntStandard)
        C_forBi = ForwardModel(E_resBi, C_resBi ,nt=ntStandard)
        E_resB.append(E_resBi)
        C_resB.append(C_resBi)
        C_forB.append(C_forBi)
    
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Kalman inversion calculations
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if run_KF:
        print 'Running the Kalman filter .....' 
        
        E_resKFi,C_resKFi,B_KF = Kalman(dataStandard, loc_obsStandard,nt=ntStandard)
        C_forKFi = ForwardModel(E_resKFi, [0]*nx ,nt=ntStandard)
        E_resKF.append(E_resKFi)
        C_resKF.append(C_resKFi)
        C_forKF.append(C_forKFi)
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Kalman ensemble
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if run_KE:
        print 'Running the Kalman ensemble......'
    
        NN = 2000
        
        E_resKEi,C_resKEi = EnsembleKalman(dataStandard, loc_obsStandard,NN,nt=ntStandard)
        C_forKEi = ForwardModel(E_resKEi, [0]*nx ,nt=ntStandard)
        E_resKE.append(E_resKEi)
        C_resKE.append(C_resKEi)
        C_forKE.append(C_forKEi)
    
    
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
    savL.append(L[0])
    L_inv, L_adj = np.linalg.inv(L), np.transpose(L)
    if precon:
        xp = state_to_precon(x0) # Preconditioned prior state
    else:
        xp = x
    
    xp_opt = sc.optimize.fmin_cg( calc_J, xp, calc_dJdx, gtol = tol , maxiter = 80)
    x_opt  = precon_to_state(xp_opt)
    E_resAi = x_opt[:nx]
    C_resAi = x_opt[nx:]
    
    C_forAi = ForwardModel(E_resAi, C_resAi, ntA)
    
    E_resA.append(E_resAi)
    C_resA.append(C_resAi)
    C_forA.append(C_forAi)
    
    # Calculate residuals
    E_residBi,  C_residBi  = resiCompleteSingle(E_resBi, C_forBi, E_true,C_real,E_prior,C_priorForward)
    E_residKFi, C_residKFi = resiCompleteSingle(E_resKFi,C_forKFi,E_true,C_real,E_prior,C_priorForward)
    E_residKEi, C_residKEi = resiCompleteSingle(E_resKEi,C_forKEi,E_true,C_real,E_prior,C_priorForward)
    E_residAi,  C_residAi  = resiCompleteSingle(E_resAi, C_forAi, E_true,C_real,E_prior,C_priorForward)
    E_residB.append(E_residBi);   C_residB.append(C_residBi)
    E_residKF.append(E_residKFi); C_residKF.append(C_residKFi)
    E_residKE.append(E_residKEi); C_residKE.append(C_residKEi)
    E_residA.append(E_residAi);   C_residA.append(C_residAi)
    
# PLOTS

# CONCENTRATIONS
x = [i*dx/1000. for i in range(nx)]
fig_con = plt.figure()
ax1 = fig_con.add_subplot(111)
ax1.plot(x, C_forB[0][-1],'b--',label='BLUE')
ax1.plot(x, C_forKE[0][-1],'c--',label = 'Kalman filter')
ax1.plot(x, C_forKF[0][-1],'m--',label = 'Kalman Ensemble')
ax1.plot(x, C_forA[0][-1],'k--', label = 'Adjoint')
ax1.plot(x, C_priorForward[-1],'r', label = 'Prior')
ax1.plot(x, C_real[ntStandard-1],'g', label = 'Exact')
for i,loc in enumerate(loc_obsA):
    ax1.plot(loc*dx/1000, dataA[-1][i],'ro',markersize = 7)
ax1.plot(loc_obsStandard*dx/1000.,[0]*len(loc_obsStandard),'bo',markersize=14,label = 'Stations')
plt.legend(loc = 'best')
ax1.set_title("Concentrations from different approaches (nt = 500, nobs = 3)")
ax1.set_xlabel("Position (in km)")
ax1.set_ylabel("Concentrations (unitless)")
ax1.legend(loc='best')
plt.savefig( 'Carcor_Concentrations')

# EMISSIONS
fig_emi = plt.figure()
ax1 = fig_emi.add_subplot(111)
ax1.plot(x, E_resB[0],label='BLUE')
ax1.plot(x, E_priors[0], label = 'Prior')
#ax1.plot(x, E_resKF,label = 'Kalman Filter')
ax1.plot(x, E_resKE[0],label = 'Kalman Ensemble')
ax1.plot(x, E_resA[0], label = 'Adjoint')
ax1.plot(x, E_true, label = 'Exact')
ax1.plot(loc_obsStandard*dx/1000.,[0]*len(loc_obsStandard),'bo',markersize=14,label = 'Stations')
ax1.set_title("Resulting emissions. nt = 500, nobs = 3")
ax1.set_xlabel("Position (in km)")
ax1.set_ylabel("Emissions (unitless)")
ax1.legend(loc='best')
plt.savefig("VarCor_Emissions")

# RESIDUALS

fig_resid = plt.figure()
ax1 = fig_resid.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_title("The effect of the correlation length on the residuals")
ax1.set_ylabel("Emission residu (% of prior)")
ax2.set_ylabel("Concentration residu (% of prior)")
ax1.set_xlabel("Correlation length (in # grid cells)")
ax1.plot(cor_len, E_residB,'r', label= "E, BLUE")
ax2.plot(cor_len, C_residB,'b', label = "C, BLUE")
ax1.plot(cor_len, E_residKE,'r--', label = "E, KE")
ax2.plot(cor_len, C_residKE,'b--', label = "C, KE")
ax1.plot(cor_len, E_residA,'r-.', label = "E, adjoint")
ax2.plot(cor_len, C_residA,'b-.', label = "C, adjoint")
ax1.legend(loc = 'center left')
ax2.legend(loc = 'center right')
plt.savefig("VarCorlength")












    
    
    
    
    
    
    
    
    