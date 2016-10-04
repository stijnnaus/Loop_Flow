# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:58:04 2016

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

ntStandard = 500   	     # Standard number of time steps for a simulation
# The definitions/routines

def smooth(data,n=3):
    '''
    Given some 1-D data, apply an n-point smoothing filter. For the first and last
    point only 2 points are used.
    '''
    L = len(data)
    data_smooth = np.empty(L)
    r = n/2 # Number of points to either side
    for i in range(L):
        high,low = i+r+1, i-r
        if low < 0:
            sm = np.mean(data[0:high])
        elif high >= n:
            sm = np.mean(data[low:])
        else:
            sm = np.mean(data[low:high])
        data_smooth[i] = sm
    return data_smooth
    
def EmissionProfile(loc_source, Ewidth, sourceStrength):
    '''
    Generates an emission profile.
    '''
    E = np.zeros(nx) # Emission field (/time step)
    
    for loc in loc_source:
        strength = (random.random() * sourceStrength)*dt
        add = random.random() # Correction so that max of source is not always in the middle of a cell
        for i in range(loc-Ewidth/2,loc+1+Ewidth):
            if i >= nx: # BC
                E[i-nx] += norm.pdf(i,loc+add)*strength
            else:
                E[i] += norm.pdf(i,loc+add)*strength
    return np.array(E)

def ForwardModel(E, Cstart,nt = ntStandard):
    '''
    The main routine for the forward model, that calculates the concentration 
    profile through time from the emission profile E.
    '''
    # Useful constants
    q = kap * dt/dx**2
    q2 = 2*q
    s = u*dt/dx
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
    
def genStations(nobs, rand = False):
    '''
    Generates nobs observation stations, either randomly or evenly
    distributed.
    '''
    
    if rand:
        random.seed(seed)
        loc_obs = np.sort(np.array([ random.randint( 0, nx - 1 ) for i in range(nobs)]))
    else:
        loc_obs = np.array([int((i+.5) * nx / nobs) for i in range(nobs)])
    return loc_obs    
    
def genData(C, loc_obs):
    '''
    Generate a perturbed dataset Cpert from C. Values at nobs stations are
    calculated over the whole time domain. The perturbed value is taken from 
    a gaussian with as sigma a certain percentage (Oerror) of the mean steady state
    concentration.
    '''
    
    ntime = len(C)
    nobs = len(loc_obs)
    
    Cpert = np.empty((nobs, ntime))
    
    for i in range(nobs):
        loc = loc_obs[i]
        for j in range(ntime):
            Cpert[i][j] =  np.random.normal(C[j][loc],Oerror)
            
    return np.array(Cpert.transpose().flatten())
      

# +++++++++++++++++++++++++++++++++++++++++++++++++++
# Inverse approach
# +++++++++++++++++++++++++++++++++++++++++++++++++++

def prior(E, var, maxshift=0):
    '''
    Determines a prior estimate based on the true state E, perturbed
    by a Gaussian with variance var and mean the true state.
    Also allows of random shifting of the E values by a random number in range
    [0,maxshift].
    '''
    
    n = len(E)
    E_prior = np.empty(n)
    
    for i in range(n):
        shift = random.randint(0, maxshift) # Random shift of the peak
        # Make a new random guess
        if i+shift < n:
            new = np.random.normal(E[i+shift],var)
        else:
            new = np.random.normal(E[n-1],var)
            
        #if new<0:
        #    new = 0
        E_prior[i] = new
    return E_prior


def BLUEAnalysis(y, loc_obs, nt = ntStandard):
    '''
    The model that retrieves the optimal starting concentration and emission
    profiles from a perturbed state x0 and the observations y.
    '''
    random.seed(seed)
    start = time.time()
    
    nobs = len(loc_obs)    
    
    # Constructing the H matrix:    
    M_empty = scs.diags([0] , [0] , [nx, nx], "csc") # MxM zeros (Top right)
    M_ident = scs.identity(nx) # MxM identity (Top left and bottom left)
    
    # Bottom right matrix
    q = kap * dt/dx**2
    q2 = 2*q
    s = u*dt/dx
    lt = lamb*dt
    b = 1 - lt - q2
    c = q - s
    a = q + s
    abc_diag = scs.diags([ [a] , [b] , [c] , [a] , [c] ], [-1 , 0 , 1 , nx-1 , -nx+1],  [nx,nx], "csc")
    
    # Combine into forward matrix
    top = scs.hstack([M_ident,M_empty])
    bot = scs.hstack([M_ident,abc_diag])
    F0 = scs.vstack([top,bot]) # For 1 time step
    
    O0 = scs.diags([0],[0], [nobs, 2*nx], "csc") # Obs operator 1 timestep
    O0 = scs.lil_matrix(O0)
    count = 0
    for loc in loc_obs:
        for i in range(nx):
            if i == loc:
                O0[count,i+nx] = 1
        count+=1
    O0 = scs.csc_matrix(O0)
    
    for i in range(nt):
        if i == 0:
            Fpow = F0
            H = O0*Fpow
        else:
            Fpow = Fpow*F0
            H = scs.vstack([H,O0*Fpow])
    HT = H.transpose()
    
    # Constructing B: Two types of errors (Emission & Concentration)
    B = scs.csc_matrix(Bmatrix)
    
    # Constructing R:
    R = scs.diags([Oerror**2],[0],[nobs*nt,nobs*nt])
    
    # So now K is easily computed:
    K = B * HT * scs.linalg.inv( ( H * B * HT + R ) )
    
    # And the best estimate:
    
    print H.toarray().shape
    xbest = x0 + K * ( y - H*x0 ) # Best estimate state vector
    covPost = scs.linalg.inv( scs.linalg.inv(B) + HT * scs.linalg.inv(R) * H ) # Best estimate posterior covariance
    Ebest = xbest[0:nx]
    Cbest = xbest[nx:]
    
    end = time.time()
    print(end - start)
    times.append(end-start)
    
    return Ebest, Cbest, covPost
    
def Kalman(y, loc_obs, nt = ntStandard):
    '''
    Inversely determines the best estimate of the a priori state, based on
    the prior state x0 and the dataset y.
    Prior emission and concentration errors are given by Error (relative) and 
    Cerror (absolute).
    Observational errors are given by Oerror (relative).
    ''' 
    random.seed(seed)
    start = time.time()
    
    nobs = len(loc_obs)   
    
    # Constructing the H matrix:    
    M_empty = scs.diags([0] , [0] , [nx, nx], "csc") # MxM zeros (Top right)
    M_ident = scs.identity(nx) # MxM identity (Top left and bottom left)
    
    # Bottom right matrix
    q = kap * dt/dx**2
    q2 = 2*q
    s = u*dt/dx
    lt = lamb*dt
    b = 1 - lt - q2
    c = q - s
    a = q + s
    abc_diag = scs.diags([ [a] , [b] , [c] , [a] , [c] ], [-1 , 0 , 1 , nx-1 , -nx+1],  [nx,nx], "csc")
    
    # Combine into forward matrix
    top = scs.hstack([M_ident,M_empty])
    bot = scs.hstack([M_ident,abc_diag])
    F0 = scs.vstack([top,bot],format="csc") # Prediction matrix for 1 time step
    F0T = F0.transpose()
    
    O0 = scs.diags([0],[0], [nobs, 2*nx], "csc") # Obs operator 1 timestep
    O0 = scs.lil_matrix(O0)
    count = 0
    for loc in loc_obs:
        for i in range(nx):
            if i == loc:
                O0[count,i+nx] = 1
        count+=1
    O0 = scs.csc_matrix(O0)
    H0 = O0*F0
    H0T = H0.transpose()
    
    # Constructing B: Two types of errors (Emission & Concentration)
    B = scs.csc_matrix(Bmatrix)
    
    # Constructing R:
    R = scs.diags([Oerror**2],[0],[nobs,nobs],"csc")
    
    # Running the loop through time
    x_prior = x0
    B_prior = B
    for i in range(nt):        
        y_t = y[nobs*i:nobs*(i+1)] # Data of current timestep
        K = B_prior * H0T * scs.linalg.inv( ( H0 * B_prior * H0T + R ) ) # Kalman gain
        
        x_post = x_prior + K * (y_t - H0 * x_prior) # Update
        B_post = B_prior - K * H0 * B_prior
        
        x_prior = F0*x_post
        B_prior = F0*B_post*F0T

    
    end = time.time()
    print end - start
    times.append(end-start)
    
    # Best estimate:
    E_best = x_prior[:nx]
    C_best = x_prior[nx:]
    return E_best,C_best, B_prior
    
def EnsembleKalman(y, loc_obs,NN , nt = ntStandard):
    '''
    The model for the ensemble Kalman approach.
    In this approach we evolve the model NN times, starting from perturbed
    x and B matrices. The perturbations are proportional to the E and C errors.
    '''
    random.seed(seed)
    start = time.time()
    
    nobs = len(loc_obs)   
    
    # Constructing the H matrix:    
    M_empty = scs.diags([0] , [0] , [nx, nx], "csc") # MxM zeros 
    M_ident = scs.identity(nx) # MxM identity 
    
    # Bottom right matrix
    b = 1 - lamb*dt - 2*kap * dt/dx**2
    c = kap * dt/dx**2 - u*dt/dx
    a = kap * dt/dx**2 + u*dt/dx
    abc_diag = scs.diags([ [a] , [b] , [c] , [a] , [c] ], [-1 , 0 , 1 , nx-1 , -nx+1],  [nx,nx], "csc")
    
    # Combine into forward matrix
    top = scs.hstack([M_ident,M_empty])
    bot = scs.hstack([M_ident,abc_diag])
    F0 = scs.vstack([top,bot],format="csc") # Prediction matrix for 1 time step
    
    O0 = scs.diags([0],[0], [nobs, 2*nx], "csc") # Obs operator 1 timestep
    O0 = scs.lil_matrix(O0)
    count = 0
    for loc in loc_obs:
        for i in range(nx):
            if i == loc:
                O0[count,i+nx] = 1
        count+=1
    O0 = scs.csc_matrix(O0)
    H0 = np.matrix((O0*F0).toarray())
    H0T = H0.transpose()
    
    # Variance vectors and R
    B = Bmatrix
    measError = np.array([Oerror**2]*nobs)
    R = np.matrix(np.diag(measError))
    
    # 1 x NN matrix filled with 1's
    e1xN = np.matrix(np.ones((1,NN)))
      
    # Monte Carlo part
    
    # Generate an ensemble X
    npr = len(x0)
    X = []
    for i in range(NN):
        xpert = np.empty(npr)
        for j in range(npr):
            xpert[j] = np.random.normal(x0[j],np.sqrt(priorError[j]))
        X.append(xpert)
    X = (np.matrix(X)).transpose()
    
    
    # Generate an ensemble data space: at every timestep an nobs x NN matrix
    Dall = [] 
    
    for i in range(nt):
        
        y_t = y[nobs*i:nobs*(i+1)]
        
        D = []
        ypert = np.empty(nobs)
        for j in range(NN):
            for m in range(nobs):
                ypert[m] = np.random.normal(y_t[m],Oerror)
            D.append(ypert)
        Dall.append(np.matrix(D).transpose())
    
    
    Xprior = X
    for i in range(nt):
        
        Xmean = np.mean(Xprior,axis = 1) # Ensemble mean state vector
        A = np.matrix(Xprior - Xmean * e1xN)
        C = ( A * A.transpose() ) / float((NN-1)) # Covariance matrix
        D = np.matrix(Dall[i])
        
        K = C * H0T * np.linalg.inv( ( H0 * C * H0T + R ) ) # Gain
        
        Xmid = Xprior + K * ( D - H0 * Xprior )
        
        Xprior = F0 * Xmid # The prior for the next time step
    
    end = time.time()
    print(end - start)
    times.append(end-start)
    
    # Best estimate:
    x_best = np.mean( Xprior,axis=1 )
    
    E_best = np.array((x_best[:nx]).transpose())
    C_best = np.array((x_best[nx:]).transpose())
    return E_best[0],C_best[0]
        
def AdjointModel(mismatch, loc_obs, nt = ntStandard):
    '''
    Uses the adjoint model to calculate dJ/dx.
    In the adjoint we split operations on C in the emissions E and all other 
    operations (advection, diffusion & radioactive decay).
    Returns HT * R^(-1) * (Hx - y) (2nx x 1: one part of dJ/dx)
    ''' 
    
    # Useful constants 
    nobs = len(loc_obs)
    b = 1 - lamb*dt - 2*kap * dt/dx**2
    c = kap * dt/dx**2 - u*dt/dx
    a = kap * dt/dx**2 + u*dt/dx
    
    # Sensitivities
    dC = np.zeros(nx)
    dE = np.zeros(nx)
    
    pulses = mismatch/Oerror**2
    
    # Run the adjoint model
    for i in range(nt):
        
        # Add the adjoint pulse:
        dC += adj_obs_oper( pulses[-(i+1)], loc_obs )
        
        # Emission step
        dE += dC
            
        # Advection/diffusion/radioactive decay step
        dC_temp = np.empty(nx)
        for m in range(nx):
            dC_temp[m] = c * dC[m-1] + b * dC[m] + a * dC[(m+1) % nx]
        dC = dC_temp.copy()
    
    return np.concatenate((dE,dC))
    

    
def obs_oper(C,loc_obs):
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
            for loc in loc_obs:
                Cobsi.append(C[i][loc])
            Cobs.append(Cobsi)
            
    # 1D array
    elif Cshape == 1:
        nx = len(C)
        for loc in loc_obs:
            Cobs.append([C[loc]])
    
    else:
        print 'Invalid array shape.'
        return 
    
    return np.array(Cobs)

def adj_obs_oper(C_obs,loc_obs):
    '''
    Maps observations (nobs x nt) to the nx grid.
    Works for 1D (1 timestep) and 2D (nt timesteps) arrays.
    Returns nt x nx array.
    '''
    C_obs = np.array(C_obs)
    Oshape = len(C_obs.shape)
    if Oshape == 1: # 1D array (1 timestep)
        Cout = np.zeros(nx)
        for m,loc in enumerate(loc_obs):
            Cout[loc] = C_obs[m]
    
    elif Oshape == 2: # 2D array (nt timesteps)
        nt = len(C_obs[0])
        Cout = np.zeros((nt,nx))
        for i in range(nt):
            for m,loc in enumerate(loc_obs):
                Cout[i][loc] = C_obs[m][i]            
    return Cout
    
def calc_mismatch(x0,y,loc_obs,nt=ntStandard):
    '''
    Calculate the mismatch between C_obs calculated from E0 and C0 forward
    in time and y (observation data).
    Returns the mismatch. (same dimensions as y,C_obs (nobs nt x 1?))
    '''
    
    E0,C0 = x0[:nx],x0[nx:]
    y = np.array(y)
    C = ForwardModel(E0, C0, nt = nt)
    C_obs = obs_oper(C,loc_obs)
    
    mismatch = C_obs - y
    
    return mismatch
    
def calc_J(mismatch,delx='Not applicable'):
    '''
    Calculates the cost function J, either including the term involving the
    deviation from the prior or (by default) not.
    '''
    Jobs = 0.5*np.sum((mismatch / Oerror) **2)
    if delx == 'Not applicable':
        return Jobs
    Jprior = 0.5 * np.dot(delx.transpose(), np.dot(B_inv, delx))
    return Jobs + Jprior

def AdjointOptimize(x0, y, loc_obs, pert = 10**-11, max_dJdx = 10**(-3)):
    '''
    The routine that uses the AdjointModel iteratively to optimize the prior.
    '''
    x0,y = np.array(x0),np.array(y)
    nobs = len(loc_obs)
    ntA = len(y.flatten()) / nobs
    
    x = x0
    continu = True
    while continu:
        
        mismatch = calc_mismatch( x, y, loc_obs, ntA )
        
        # Calculating dJ/dx
        adjoint = AdjointModel( mismatch, loc_obs, ntA )
        delx = x-x0
        dJdx = np.dot(B_inv, delx) + adjoint 
        dJdx_sum = np.sum(abs(dJdx))
        
        # Calculating J
        J = calc_J(mismatch, delx)
        
        x_opt = sc.optimize.fmin_bgsf()
        
        print delx
        print J, np.max(abs(dJdx)), dJdx_sum
        
        if dJdx_sum < max_dJdx:
            continu = False
        
    return x

# RESIDUALS

def resiCalc(A_est, A_exact):
    '''
    Computes the sum of the square of the differences between some estimate 
    1 or 2D array A_est and A_exact: We call this the residual
    '''
    A_est, A_exact = np.array(A_est), np.array(A_exact)
    Dim = len(A_est.shape) # Check dimensions
    
    if Dim != 1 and Dim != 2:
        return 'That was not 1D or 2D!'
        
    residu = np.sum( ( A_est - A_exact )**2 )
    
    if Dim == 2:
        return residu / len(A_est)
    
    return residu
        
def resiComplete(E_est,C_est,E_exact,C_exact,E_prior,C_prior):
    '''
    Computes a set of Emission residuals and Concentration residuals, relative
    to the prior residuals. Returns to 1D arrays, 1 for E, 1 for C, with each
    number in an array corresponding to some test run (e.g. varying timestep).
    '''
    resEPrior = resiCalc(E_prior,E_exact)
    L = len(E_est) # Number of runs
    residuE = []
    residuC = []
    for i in range(L):
        nt = len(C_est[i]) # Number of time steps
        resE = resiCalc(E_est[i],E_exact)
        resC = resiCalc(C_est[i], C_exact[:nt])
        resCPrior = resiCalc(C_prior[:nt],C_exact[:nt])
        residuE.append(100 * resE/resEPrior)
        residuC.append(100 * resC/resCPrior)
    return np.array(residuE), np.array(residuC)
    

