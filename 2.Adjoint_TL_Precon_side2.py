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

ntStandard = 2000

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
            if m == nx-1:
                C[n+1][m] = E[m] + (1 - lt - q2) * C[n][m] + ( q - s ) * C[n][0] + ( q + s ) * C[n][m-1]
            else:
                C[n+1][m] = E[m] + (1 - lt - q2) * C[n][m] + ( q - s ) * C[n][m+1] + ( q + s ) * C[n][m-1]
    
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

def calc_JTL(xp):
    '''
    Calculates the cost function J from the guess x0.
    '''
    x0 = precon_to_state(xp)
    mismatch,local = calc_mismatchTL(x0)
    delx = x0 - x_priorA
    Jobs = 0.5*np.sum((mismatch / Oerror) **2)
    Jprior = 0.5 * np.dot(delx, np.dot(B_inv, delx))
    J_obs.append(Jobs)
    J_prior.append(J_prior)
    #print 'Jobs, Jprior = ',Jobs,Jprior
    #print 'x0: ', np.sum(abs(x0))
    print 'J:',(Jobs + Jprior)*reduction
    return (Jobs + Jprior)*reduction
    
def calc_dJdxTL(xp):
    '''
    Computes derivative dJ/dx from the guess x0
    '''
    x0 = precon_to_state(xp)
    mismatch,Csaved = calc_mismatchTL(x0)
    u_old = x0[-1]
    adjoint = AdjointModelTL( mismatch , Csaved, u_old )
    delx = x0 - x_priorA
    dJdx = np.dot(B_inv, delx) + adjoint 
    #print 'dJdx (adjoint, prior) =',np.sum(adjoint),np.sum(np.dot(B_inv,delx))
    dJdxp = np.dot(L_adj, dJdx)
    #print 'dJdx (adjoint, prior) =',np.sum(adjoint),np.sum(np.dot(B_inv,delx))
    print 'deriv:',max(abs(dJdxp))*reduction
    return dJdxp*reduction
    
def tostate(E,C,u):
    xa = np.concatenate((E,C))
    x = np.append(xa,u)
    return x

def preConTL(B_start, corrE,corrC):
    '''
    Preconditioning by applying a spatial correlation corr in matrix B.
    '''
    B_start = np.array(B_start)
    empty = np.zeros((nx,nx))
    
    E_precon = empty.copy()
    C_precon = empty.copy()
    errors = np.diag(B_start)
    E_errors = errors[:nx]
    C_errors = errors[nx:-1] 
    U_error  = errors[-1]
    
    for i in range(nx):
        for j in range(nx):
            if i == j:
                E_precon[i][i] = E_errors[i]
                C_precon[i][i] = C_errors[i]
            else:
                posi = abs(i-j)
                corrEi = max(corrE[posi-1],corrE[nx-posi-1]) # BC emi
                corrCi = max(corrC[posi-1],corrC[nx-posi-1]) # BC emi
                E_precon[i][j] = corrEi*Eerror**2
                C_precon[i][j] = corrCi*Cerror**2
    # Putting them together
    B_top = np.hstack((E_precon,empty))
    B_bot = np.hstack((empty,C_precon))
    B_noU = np.vstack((B_top,B_bot))
    # Adding U
    B_final = np.zeros((2*nx+1,2*nx+1))
    B_final[:-1,:-1] = B_noU
    B_final[-1,-1] = U_error
    
    return np.matrix(B_final)

def state_to_precon(x):
    '''
    Convert the state and derivative to the preconditioned space.
    '''
    return np.dot( L_inv, (x - x_priorA) ) 

def precon_to_state(xp):
    '''
    Convert the preconditioned state to the original space.
    '''
    return np.dot( L, xp ) + x_priorA

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initialization
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
seed = 1
times = []
random.seed(seed)

ustarts = [0.0,6.0,8.0,10.0,12.0]
experiment = 'VarUstart'
uposts,E_resid,C_resid = [],[],[]
for ustart in ustarts:
    for ti in range(20):
        # ------------------------------------------------------------------------------
        # Parameters that are fixed through each individual run
        # Simulation parameters
        precon = True               # Preconditioning on or off
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
        Eerror = 1e-1 * max(E_true)          # emission error
        Cerror = 5e-8 # prior error
        Oerror = 1e-3 # observation error as used in the inversions
        Uerror = 5.
        Oerror_real = Oerror # the actual imposed error
        # Construct prior error (/variance) matrix B, possibly with preconditioning
        priorError = np.array([Eerror**2]*nx+[Cerror**2]*nx + [Uerror**2])
        Bmatrix = np.matrix( np.diag(priorError) )
        cor_lenE = 5.
        cor_lenC = 5.
        corrE = [np.exp(-((i/cor_lenE))) for i in range(1,nx+1)]
        corrC = [np.exp(-((i/cor_lenC))) for i in range(1,nx+1)]
        Bmatrix = preConTL( Bmatrix,corrE,corrC )
        Bmatrix_inv = np.linalg.inv(Bmatrix)
        B_inv = np.array(Bmatrix_inv)
        
        
        nobsStandard = 3 # Number of measurement stations
        loc_obsStandard = genStations(nobsStandard, rand = False) # Standard locations meas stations
        # Generating input data
        data = genData(C_real, loc_obsStandard) # Observations 
        x0 = np.random.multivariate_normal(xtrue,Bmatrix) # Prior state
        x0[-1] = ustart
        C_priorForward = ForwardModelTL(x0,nt = ntMax) # Perturbed model data
        
        
        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Adjoint TL model
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        print 'Running the Adjoint Tangent Linear model......'
        # ---------------------------------------------------------------
        # STANDARD RUN
        
        L = np.sqrt( np.array(Bmatrix) )
        L_inv,L_adj = np.linalg.inv(L), np.transpose(L)
        
        tol = 1e-5
        loc_obsA = loc_obsStandard
        nobsA = len(loc_obsA)
        ntA = ntStandard
        x_priorA = x0.copy()
        dataA = np.array( np.split( data[:nobsA*ntA], ntA ) )
        
        x0p = state_to_precon(x0)
        x_optp = sc.optimize.fmin_cg(calc_JTL, x0p, calc_dJdxTL,gtol = 1e-3, retall = True, disp = True, maxiter = 80)
        x_opt = precon_to_state(x_optp[0])
        E_resA = x_opt[:nx]
        C_resA = x_opt[nx:-1]
        U_resA = x_opt[-1]
        
        C_forwA = ForwardModelTL(x_opt, ntA)
        
        #for i,x in enumerate(x_opt[1]):
        #    if i%3 == 0:
        #        plt.plot(x[:-1],label = 'fit' + str(i))
        #plt.figure()
        #plt.plot(x_opt[:-1], label = 'Final')
        #plt.plot(x_priorA[:-1], label = 'Prior')
        #plt.plot(xtrue[:-1])
        #plt.legend(loc = 'best')
        uposts.append(x_opt[-1])
        resis = resiCompleteSingle(E_resA, C_forwA, E_true,C_real,x0[:nx],C_priorForward)
        E_resid.append(resis[0])
        C_resid.append(resis[1])
        print 'Iteration:',len(uposts)
    print 'end of:',ustart


#starts = [0.1,5.5,8.0,10.0,12.0]
#means = [np.mean(uposts1),np.mean(uposts1),np.mean(uposts2),np.mean(uposts3),np.mean(uposts4)]
#errors = [np.std(uposts1),np.std(uposts1),np.std(uposts2),np.std(uposts3),np.std(uposts4)]
#E_resid = [E_residA0,E_residA1]
#C_resid = [C_residA0,C_residA1]
uposts = np.array(np.split(np.array(uposts.flatten()),5))
umeans, ustds = np.mean(uposts,axis=1),np.std(uposts,axis=1)
plt.title('Influence of Uprior, all else being equal (Uerror = 5)')
plt.errorbar(ustarts,umeans,yerr = ustds, fmt = 'o')
plt.axis([-.5,12.5,4,12.5])
plt.savefig('AdjTL_VarUstart')


'''
x = [i*dx/1000. for i in range(nx)]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(x, C_priorForward[ntStandard - 1], label = 'Prior')
ax1.plot(x, C_forwA[ntStandard - 1], label = 'Adjoint')
ax1.plot(x, C_real[ntStandard - 1], label = 'Exact')
for i,loc in enumerate(loc_obsA):
    ax1.plot(loc*dx/1000, dataA[-1][i],'ro',markersize = 7)
ax1.set_title("Concentrations Adjoint Tangent Linear \n(upri = "+str(round(x0[-1],2))+", upost = "+str(round(x_opt[-1],2))+")")
ax1.set_xlabel("Position (in km)")
ax1.set_ylabel("Concentrations (unitless)")
ax1.legend(loc='best')
plt.savefig('AdjTL_Pre_Concentrations_'+experiment)

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
plt.savefig('AdjTL_Pre_Emissions_'+experiment)
'''
print 'Prior u: ',x0[-1]
print 'Posterior u: ',x_opt[-1]