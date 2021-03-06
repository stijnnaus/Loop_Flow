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
KEextensive = False
long_runs = False # If False, only runs the methods for the standard settings

# ------------------------------------------------------------------------------
# Parameters that are fixed through each individual run
# Simulation parameters
precon = True # Preconditioning on or off
(nx,dx) = (int(1e2), 2e4)   # nx grid cells; grid size (m)
ntMax = 30000               # Maximum number of timesteps
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
Ewidth = 3                 # Peak width for each (Gaussian) source
maxStrength = 1e-3          # Max source strength
# Inverse model parameters
E_true = EmissionProfile(loc_source,Ewidth,maxStrength) # True emission profile
Cstart = np.zeros(nx) # Starting concentration profile (Just 0 everywhere)  
xtrue = np.concatenate((E_true,Cstart))
C_real = ForwardModel(E_true, Cstart, nt = ntMax) # The exact model data
C_steady = ForwardModel(E_true, Cstart, nt = ntSteady) # Steady state model data
# Errors
Eerror = 0.1 * max(E_true)          # emission error
Cerror = 5e-3 # prior error
Oerror = 1e-2 # observation error as used in the inversions
Oerror_real = Oerror # the actual imposed error
# Construct prior error (/variance) matrix B, possibly with preconditioning
priorError = np.array([Eerror**2]*nx+[Cerror**2]*nx)
Bmatrix = np.matrix( np.diag(priorError) )
if precon: # Preconditioning
    cmax = .75
    sigE = 4.
    sigC = 4.
    corrE = [cmax * np.exp(-((i/sigE)**2)) for i in range(nx)]
    corrC = [cmax * np.exp(-((i/sigC)**2)) for i in range(nx)]
    Bmatrix = preCon(Bmatrix,corrE,corrC)
Bmatrix_inv = np.linalg.inv(Bmatrix)
B_inv = np.array(Bmatrix_inv)



nobsStandard = 3 # Number of measurement stations
loc_obsStandard = genStations(nobsStandard, rand = False) # Standard locations meas stations
# Generating input data
data = genData(C_real, loc_obsStandard) # Observations 
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
    
    # ----------------------------------------------------------
    # Studying the effects of the length of the simulation
    
    E_resT = [] 
    C_resT = []
    covPostT = []
    ntT = [100,300,500]
    
    for nti in ntT:
        data_sel = data[:nti*nobsStandard]
        res = BLUEAnalysis(data_sel, loc_obsStandard,nt = nti)
        
        E_resT.append(res[0])
        C_resT.append(res[1])
        covPostT.append(res[2].toarray())
    
    # Using the best estimate for forward modelling
    C_forwT = []
    for i,nti in enumerate(ntT):
        
        CresT = ForwardModel(E_resT[i], C_resT[i],nt = nti)
        C_forwT.append(CresT)
    
    # Studying the effect when stations are distributed randomly
    if randomrun:
        # Generating observations
        loc_obsTR = genStations(nobsStandard, rand = True)
        dataTR = genData(C_real, loc_obsTR)
        
        E_resTR = []
        C_resTR = []
        covPostTR = [],
        for nt in ntT:
            data_sel = dataTR[:nt*nobsStandard]
            
            res = BLUEAnalysis(data_sel, loc_obsTR)
            
            E_resTR.append(res[0])
            C_resTR.append(res[1])
            covPostTR.append(res[2].toarray())
        
        # Using the best estimate for forward modelling
        C_forwTR = []
        for i,nt in enumerate(ntT):
            CresTR = ForwardModel(E_resTR[i], C_resTR[i]) 
            C_forwTR.append(CresTR)
        
    # ----------------------------------------------------------
    # Studying the effect of varying number of obs stations
    # Generating observations
    nobsO = [2,3,4]
    C_priorO = ForwardModel(E_prior,C0_prior)
    
    E_resO = [] # Even distribution
    C_resO = []
    covPostO = []
    E_resOR = [] # Random distribution
    C_resOR = []
    covPostOR = []
    for nobs in nobsO:
        loc_obsO = genStations(nobs, rand = False)
        dataO = genData(C_real, loc_obsO)
        dataO = dataO[:nobs*ntStandard]
    
        resO  = BLUEAnalysis(dataO, loc_obsO)
        
        E_resO.append(resO[0])
        C_resO.append(resO[1])
        covPostO.append(res[2].toarray())
        
        if randomrun:
            loc_obsOR = genStations(nobs, rand = True)
            dataOR = genData(C_real, loc_obsOR)
            dataOR = dataOR[:nobs*ntStandard]
            
            resOR = BLUEAnalysis(dataOR, loc_obsOR)
            
            E_resOR.append(resOR[0])
            C_resOR.append(resOR[1])
            covPostOR.append(res[2].toarray())
        
    # Using the best estimate for forward modelling
    C_forwO = []
    C_forwOR = []
    for i in range(len(nobsO)):
        CresO = ForwardModel(E_resO[i], C_resO[i])
        C_forwO.append(CresO)
        
        if randomrun:
            CresOR = ForwardModel(E_resOR[i], C_resOR[i])
            C_forwOR.append(CresOR)



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Kalman inversion calculations
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if run_KF:
    print 'Running the Kalman filter .....' 
    # ----------------------------------------------------------
    # Studying the effects of the length of the simulation
    
    E_resTK = []
    C_resTK = []
    covPostTK = []
    C_priorT = ForwardModel(E_prior,C0_prior,nt=ntMax)
    
    for steps in ntT:
        data_sel = data[:steps*nobsStandard]
        res = Kalman(data_sel, loc_obsStandard, nt = steps)
        E_resTK.append(res[0])
        C_resTK.append(res[1])
        covPostTK.append(res[2].toarray())
    
    # Using the best estimate for forward modelling
    C_forwTK = []
    for i,nti in enumerate(ntT):
        CresTK = ForwardModel(E_resTK[i], C_resTK[i], nt=nti)
        C_forwTK.append(CresTK)
    
    # Studying the effect when stations are distributed randomly
    
    if randomrun:
        # Generating observations
        loc_obsTRK = genStations(nobsStandard, rand = True)
        dataTRK = genData(C_real, loc_obsTR)
        
        E_resTRK = []
        C_resTRK = []
        covPostTRK = []
        for steps in ntT:
            data_sel = dataTRK[:steps*nobsStandard]
            res = Kalman(data_sel, loc_obsTRK)
            E_resTRK.append(res[0])
            C_resTRK.append(res[1])
            covPostTRK.append(res[2].toarray())
            
        # Using the best estimate for forward modelling
        C_forwTRK = []
        for i in range(len(ntT)):
            CresTRK = ForwardModel(nx,dx ,ntT[i],dt, kap, lamb, u, E_resTRK[i], C_resTRK[i]) 
            C_forwTRK.append(CresTRK)
    
    
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Kalman ensemble
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if run_KE:
    print 'Running the Kalman ensemble......'
    
    # Varying ensemble size
    # ----------------------------------------------------------
    # Initialization
    nobsKE = nobsStandard
    loc_obsKE = genStations(nobsKE) # Observation stations
    ntK = ntStandard # Time steps
    dataKE = genData(C_real[:ntK], loc_obsKE) # Perturbed dataset
    NN = [50,150,500,1000] # Ensemble sizes to be tested
    E_resKE = [] # Emission esults
    C_resKE = []
    
    # Ensemble runs
    for NNi in NN:
        Ei, Ci = EnsembleKalman(dataKE, loc_obsKE,NNi,nt=ntK)
        E_resKE.append(Ei)
        C_resKE.append(Ci)
        
    E_resKE, C_resKE = np.array(E_resKE), np.array(C_resKE)
    
    C_forwKE = []
    for i in range(len(NN)):
        CresKE = ForwardModel(E_resKE[i], np.array([0.]*nx),nt=ntK)
        C_forwKE.append(CresKE)
    C_forwKE = np.array(C_forwKE)
    
    # Varying number of timesteps
    # ----------------------------------------------------------
    NNfixed = 2000
    ntTKE = ntT + [2000] 
    
    E_resTKE = []
    C_resTKE = []
    for steps in ntTKE:
        data_sel = data[:steps*nobsStandard]
        
        Ei,Ci = EnsembleKalman(data_sel, loc_obsKE,NNfixed, nt = steps)
        
        E_resTKE.append(Ei)
        C_resTKE.append(Ci)
        
    E_resTKE, C_resTKE = np.array(E_resTKE), np.array(C_resTKE)
    
    C_forwTKE = []
    for i,nti in enumerate(ntTKE):
        CresTKE = ForwardModel(E_resTKE[i], np.array([0.]*nx),nt = nti)
        C_forwTKE.append(CresTKE)
    C_forwTKE = np.array(C_forwTKE)
    
    # Varying number of observation stations
    # ----------------------------------------------------------
    
    nobsOKE = nobsO + [10,15,30,50,70,90,100]
    ntOKE = ntStandard
    C_priorOKE = ForwardModel(E_prior,C0_prior, nt = ntOKE)
    
    E_resOKE = [] # Emission results
    C_resOKE = [] # Concentration results
    
    for nobsi in nobsOKE:
        loc_obsOKE = genStations(nobsi, rand = False)
        dataOKE = genData(C_real, loc_obsOKE)
        dataOKE = dataOKE[:nobsi*ntOKE]
    
        Ei, Ci  = EnsembleKalman(dataOKE, loc_obsOKE,NNfixed, nt = ntOKE)
        
        E_resOKE.append(Ei)
        C_resOKE.append(Ci)
    
    # Using the best estimate for forward modelling
    C_forwOKE = []
    for i in range(len(nobsOKE)):
        CresOKE = ForwardModel(E_resOKE[i], np.array([0.]*nx), nt=ntOKE )
        C_forwOKE.append(CresOKE)
    
# Extensive KE run

if KEextensive:

    NNext = [10+i*10 for i in range(0,29)]
    NNext += [max(NNext) + i*30 for i in range(1,30)]
    NNext += [max(NNext) + i*80 for i in range(1,25)]
    E_resKEext=[]
    C_resKEext=[]
    for NNi in NNext:
        Ei, Ci = EnsembleKalman(dataKE, loc_obsKE,NNi, nt = ntK)
        E_resKEext.append(Ei)
        C_resKEext.append(Ci)
        if NNi%100 == 0:
            print NNi
    
    E_resKEext, C_resKEext = np.array(E_resKEext), np.array(C_resKEext)
    
    C_forwKEext = []
    for i in range(len(NNext)):
        CresKEext = ForwardModel(E_resKEext[i], np.array([0.]*nx), nt = ntK)
        C_forwKEext.append(CresKEext)
    C_forwKEext = np.array(C_forwKEext)
    
    
    residuTE, residuTC = resiComplete(E_resKEext,C_forwKEext,E_true,C_real,E_prior,C_priorForward)
    
    fig_ext = plt.figure()
    ax1 = fig_ext.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.set_title('The effect on the ensemble size on the residuals.\n nt = 500; nobs = 3')
    ax1.set_ylabel("Emission residu (% of prior)")
    ax2.set_ylabel("Concentration residu (% of prior)")
    ax1.plot(NNext[10:],smooth(residuTE[10:]),'r',label='E')
    ax2.plot(NNext[10:],smooth(residuTC[10:]),'b',label='C')
    ax1.legend(loc=1)
    ax2.legend(loc=2)
    plt.savefig('Extensive_ensemble_run_residuals')


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Adjoint model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print 'Running the Adjoint model......'
# ---------------------------------------------------------------
# STANDARD RUN

tol = 1e-5
loc_obsA = loc_obsStandard
nobsA = len(loc_obsA)
ntA = ntStandard
x_priorA = x0
dataA = np.array( np.split( data[:nobsA*ntA], ntA ) )

x_opt = sc.optimize.fmin_cg(calc_J, x0, calc_dJdx,gtol = tol, retall = True, disp = True, epsilon = 1e-30)
E_resA = x_opt[0][:nx]
C_resA = x_opt[0][nx:]

C_forwA = ForwardModel(E_resA, C_resA, ntA)

#plt.figure()
#for i,x in enumerate(x_opt[1]):
#    if i%3 == 0:
#        plt.plot(x,label = 'fit' + str(i))
#plt.plot(x_opt[0], label = 'Final')
#plt.plot(x_priorA, label = 'Prior')
#plt.legend(loc = 'best')
    
plt.figure()
plt.plot(J_obs,label='obs')
plt.plot(J_prior,label='prior')
plt.legend()

# ---------------------------------------------------------------
# VARYING TIMESTEP

ntAT = [100,300,500,700,900,1100,2000]
E_resAT,C_resAT,C_forwAT,x_optAT,J_AT = [],[],[],[],[]

for i,steps in enumerate(ntAT):
    start = time.time()
    
    ntA = steps
    dataA = np.array( np.split( data[:nobsA*ntA], ntA ) )
    
    x_temp = sc.optimize.fmin_bfgs( calc_J, x0, calc_dJdx,gtol = tol , retall = True, disp = True )
    E_resAT.append( x_temp[0][:nx] )
    C_resAT.append( x_temp[0][nx:] )
    x_optAT.append( x_temp[0] )
    J_AT.append( calc_J( x_temp[0] ) )
    
    C_forwAT.append( ForwardModel(E_resAT[i], C_resAT[i], ntA) )
    end = time.time()
    print end-start
    plt.plot(x_temp)
    
E_resAT,C_resAT,C_forwAT = np.array(E_resAT),np.array(C_resAT),np.array(C_forwAT)
    
# ---------------------------------------------------------------
# VARYING NOBS

ntA = 500
nobsAO = [2,3,4,5,8,15,40,80,99]
E_resAO,C_resAO,C_forwAO,x_optAO,J_AO = [],[],[],[],[]

for i,stations in enumerate(nobsAO):
    start = time.time()
    
    nobsA = stations
    loc_obsA = genStations(nobsA)
    dataA = np.array( np.split( genData(C_real, loc_obsA, ntA), ntA ) )
    
    x_temp = sc.optimize.fmin_bfgs( calc_J, x0, calc_dJdx,gtol = tol , retall = True, disp = True)
    E_resAO.append( x_temp[0][:nx] )
    C_resAO.append( x_temp[0][nx:] )
    x_optAO.append( x_temp[0] )
    J_AO.append( calc_J( x_temp[0] ) )
    
    C_forwAO.append( ForwardModel(E_resAO[i], C_resAO[i], ntA) )
    
    end = time.time()
    print end-start
    
    
E_resAO,C_resAO,C_forwAO = np.array(E_resAO),np.array(C_resAO),np.array(C_forwAO)
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    