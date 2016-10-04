# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 16:04:08 2016

@author: Stijn
"""

import numpy as np
import matplotlib.pyplot as plt
import math as M
from scipy.stats import norm
import random as R
import scipy.sparse as scs
import time
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++ PLOTTING ++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

plt.figure()
plt.plot(C_resA)
plt.plot(C0_prior)

# General definitions

add = '_Add_Adjoint' # An addition to plot names to describe the experiment
rcParams.update({'figure.autolayout': True})
markertype = ['b-','b-.','b--','k-','k-.','k--','c-','c-.','c--','r-','r-.','r--']
x = [i*dx/1000. for i in range(nx)]
ntTMax = max(ntT)
ntTMin = min(ntT)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PLOTS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Steady state
fig0 = plt.figure()
ax1 = fig0.add_subplot(111)
ax1.set_title('Time it takes different stations to reach steady state')
for j,loc in enumerate(loc_obsStandard):
    ax1.plot([C_steady[i][loc] for i in range(len(C_steady))],markertype[3*j],label = 'Obs station'+str(loc))
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Concentration')
ax1.legend(loc='best')
plt.savefig('Steady_C')

fig0b = plt.figure()
ax1 = fig0b.add_subplot(111)
ax1.set_title('Difference in C at different stations between two subsequent timesteps')
for j in loc_obsStandard:
    ax1.plot([C_steady[i][j]-C_steady[i-1][j] for i in range(1,len(C_steady))])
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Change in C')
plt.savefig('Steady_Delta_C')
# -----------------------------------------------------------
# Concentrations and emissions results

# Varying timestep

# CONCENTRATIONS AT T = 500
fig_con = plt.figure()
ax1 = fig_con.add_subplot(111)
ax1.plot(x, C_forwT[2][-1],label='Even, BLUE')
ax1.plot(x, C_priorForward[499], label = 'Prior')
ax1.plot(x, C_forwA[499], label = 'Adjoint')
ax1.plot(x, C_real[499], label = 'Exact')
ax1.plot(x,C_resKE[-1],label = 'KE (size: '+str(NN[-1]) + ')')
ax1.plot(loc_obsStandard*dx/1000.,[0]*len(loc_obsStandard),'bo',markersize=14,label = 'Stations')

ax1.set_title("Concentrations from different approaches (nt = 500, nobs = 3)")
ax1.set_xlabel("Position (in km)")
ax1.set_ylabel("Concentrations (unitless)")
ax1.legend(loc='best')
plt.savefig( 'Concentrations_t500'+add)

# EMISSIONS
fig_emi = plt.figure()
ax1 = fig_emi.add_subplot(111)

ax1.plot(x, E_resT[2],label='Even, BLUE')
ax1.plot(x, E_prior, label = 'Prior')
ax1.plot(x, E_true, label = 'Exact')
ax1.plot(x, E_resKE[-1],label = 'Ensemble size:'+str(NN[-1]))
ax1.plot(x, E_resA, label = 'Adjoint')

ax1.set_title("Resulting emissions. nt = 500, nobs = 3")
ax1.set_xlabel("Position (in km)")
ax1.set_ylabel("Emissions (unitless)")
ax1.legend(loc='best')
plt.savefig('Emissions_t500'+add)

# Concentrations per station

fig_emiB = plt.figure()
ax1 = fig_emiB.add_subplot(111)
for n,j in enumerate(loc_obsStandard):
    ax1.plot( [C_forwT[2][i][j] for i in range(len(C_forwT[2]))], markertype[n], label='BLUE at ' + str(j) )
    ax1.plot( [C_forwOKE[1][i][j] for i in range(len(C_forwT[2]))], markertype[n+3], label='KE at ' + str(j) )
    ax1.plot( [C_real[i][j] for i in range(len(C_forwT[2]))], markertype[n+6], label='Real at ' + str(j) )
    ax1.plot( [C_forwA[i][j] for i in range(len(C_forwT[2]))], markertype[n+9], label='Adjoint at ' + str(j))
ax1.set_xlabel('Time step')
ax1.set_ylabel('Concentration (unitless)')
ax1.legend(loc='best')

# Residuals
residuTE, residuTC = resiComplete(E_resT,C_forwT,E_true,C_real,E_prior,C_priorForward)
residuTKE, residuTKC = resiComplete(E_resTKE, C_forwTKE, E_true, C_real, E_prior, C_priorForward)
residuATE, residuATC = resiComplete(E_resAT, C_forwAT, E_true, C_real, E_prior, C_priorForward)
#residuATE2, residuATC2 = resiComplete(E_resAT2, C_forwAT2, E_true, C_real, E_prior, C_priorForward)

if randomrun:
    residuTRE, residuTRC = resiComplete(E_resTR,C_forwTR,E_true,C_real,E_prior,C_priorForward)

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_title("The effect of the length of the simulation\n on the residuals (nobs = "+str(len(loc_obsStandard))+")")
ax1.plot(ntT, residuTE,'r', label= "E, BLUE")
ax2.plot(ntT, residuTC,'b', label = "C, BLUE")
ax1.plot(ntT, residuTKE[:len(ntT)],'r--', label = "E, KE (size: "+str(NNfixed)+")")
ax2.plot(ntT, residuTKC[:len(ntT)],'b--', label = "C, KE (size: "+str(NNfixed)+")")
ax1.plot(ntT, residuATE[:len(ntT)],'r-.', label = "E, adjoint")
ax2.plot(ntT, residuATC[:len(ntT)],'b-.', label = "C, adjoint")
if randomrun:
    ax1.plot(ntT, residuTRE,'r', label = "E, random, BLUE")
    ax2.plot(ntT, residuTRC,'b', label = "C, random, BLUE")
ax1.set_ylabel("Emission residu (% of prior)")
ax2.set_ylabel("Concentration residu (% of prior)")
ax1.set_xlabel("Number of timesteps for optimization")
ax1.legend(loc='center left')
ax2.legend(loc = 'center right')
plt.savefig("VarTime_Opt"+add)

figg = plt.figure()
ax1 = figg.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_title('The effect of length of simulation on the\n residuals in the KE (N = 300) and adjoint approach')
ax1.plot(ntTKE[1:],residuTKE[1:],'r',label = 'E, KE')
ax2.plot(ntTKE[1:],residuTKC[1:],'b',label = 'C, KE')
ax1.plot(ntAT,residuATE,'r--',label = 'E, adjoint')
ax2.plot(ntAT,residuATC,'b--',label = 'C, adjoint')
ax1.legend(loc=1)
ax2.legend(loc=2)
ax1.set_ylabel("Emission residu (% of prior)")
ax2.set_ylabel("Concentration residu (% of prior)")
plt.savefig('VarTime_KE'+add)

# Varying number of observation stations (500 time steps)

residuOE, residuOC = resiComplete(E_resO,C_forwO,E_true,C_real,E_prior,C_priorForward)
residuOKE, residuOKC = resiComplete(E_resOKE, C_forwOKE, E_true, C_real, E_prior, C_priorForward)
residuAOE, residuAOC = resiComplete(E_resAO, C_forwAO, E_true, C_real, E_prior, C_priorForward)



fig3 = plt.figure()
ax1 = fig3.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_title("The effect of the number of measurement\n stations on the quality of the optimization (nt = 500)")
ax1.plot(nobsO, residuOE, 'r', label = 'E, BLUE')
ax2.plot(nobsO,residuOC, 'b', label = 'C, BLUE')
ax1.plot(nobsO, residuOKE[:len(nobsO)], 'r--', label = 'E, KE')
ax2.plot(nobsO, residuOKC[:len(nobsO)], 'b--', label = 'C, KE')
ax1.plot(nobsO, residuAOE[:len(nobsO)], 'r-.', label = 'E, adjoint')
ax2.plot(nobsO, residuAOC[:len(nobsO)], 'b-.', label = 'C, adjoint')
if randomrun:
    ax1.plot(nobsO, residuOR, 'b--', label = 'E, random')
    ax2.plot(nobsO, residuOC, 'b--', label = 'C, random')
ax1.set_ylabel("Emission residu (% of prior)")
ax2.set_ylabel("Concentration residu (% of prior)")
ax1.set_xlabel("Number of measurement stations")
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
plt.savefig("VarObs_Opt"+add)

figg = plt.figure()
ax1 = figg.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_title("The effect of the number of measurement stations on the\n residuals in the KE (N = 300) and adjoint approach")
ax1.plot(nobsOKE[2:],residuOKE[2:],'r',label = 'E, KE')
ax2.plot(nobsOKE[2:],residuOKC[2:],'b',label = 'C, KE')
ax1.plot(nobsAO, residuAOE,'r--',label = 'E, adjoint')
ax2.plot(nobsAO, residuAOC,'b--',label = 'C, adjoint')
ax1.set_ylabel("Emission residu (% of prior)")
ax2.set_ylabel("Concentration residu (% of prior)")
ax1.legend(loc = 2)
ax2.legend(loc= 1)
plt.savefig('VarObs_KE'+add)



'''
# -----------------------------------------------------
# Kalman filter

figu0 = plt.figure()
ax1 = figu0.add_subplot(111)
ax2 = ax1.twinx()
ax1.plot(E_resTK[-1],'r-', label = "E Kalman even")
ax2.plot(C_resTK[-1],'r--', label = "C Kalman even")
ax2.plot(C_real[ntTMax],'g--',label="C Real")
if randomrun:
    ax1.plot(E_resTRK[-1], label = "Kalman random")
ax1.plot(E_prior,label = "E Prior")
ax1.plot(E, label = "E Exact")
ax1.legend(loc='best')


# -----------------------------------------------------
# Kalman filter vs analytical

residuTK = []
residuTRK = []

for i in range(len(ntT)):
    resTK = np.sum((E_resTK[i] - E)**2)    
    residuTK.append(resTK / res_prior)
    
    if randomrun:
        resTRK = np.sum((E_resTRK[i] - E)**2)
        residuTRK.append(resTRK / res_prior)

figu1 = plt.figure()
ax1 = figu1.add_subplot(111)
ax1.set_title("Difference Kalman and BLUE")
ax1.plot(ntT, residu,'r', label= "Even obs distribution (inv)")
ax1.plot(ntT, residuTK,'r--', label= "Even obs distribution (kal)")
if randomrun:
    ax1.plot(ntT, residuTRK,'b--', label = "Random obs distribution (kal)")
    ax1.plot(ntT, residuR,'b', label = "Random obs distribution(inv)")

ax1.set_ylabel("Emission residu (% of prior)")
ax1.set_xlabel("Number of timesteps for optimization")

ax1.legend(bbox_to_anchor=(1.7, 1.00))
plt.savefig("Kal_v_Ana.png")
'''

# ------------------------------------------------------------------
# Kalman ensemble

# EMISSIONS
plt.figure()
for i in range(len(NN)):
    plt.plot(x,E_resKE[i],markertype[i],label = 'Ensemble size:'+str(NN[i]))
plt.plot(x,E_true,'--',label='Real')
plt.plot(x,E_prior,'-.', label = 'Prior')
plt.title('The effect of the Kalman Ensemble size on emissions')
plt.ylabel('Emissions (unitless)')
plt.xlabel('Position (km)')
plt.legend(loc='best')

# CONCENTRATIONS

plt.figure()
for i in range(len(NN)):
    plt.plot(x,C_forwKE[i][-1],markertype[i],label = 'Ensemble size:'+str(NN[i]))
plt.plot(x,C_real[ntK-1],'g-',label = 'Real Concentration')
plt.plot(x,C_priorForward[ntK-1],'r-',label = 'Prior Concentration')
plt.title('The effect of the Kalman Ensemble size on concentrations')
plt.xlabel('Position (km)')
plt.ylabel('Concentrations (unitless)')
plt.legend(loc='best')

# Residuals

residuEKE, residuCKE = resiComplete(E_resKE,C_forwKE,E_true,C_real,E_prior,C_priorForward)

fig_res = plt.figure()
ax1 = fig_res.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_title('Residuals of the KE approach, relative to the prior\n nobs = '+str(nobsKE)+', nt = '+str(ntK))
ax1.plot(NN,residuEKE,'r',label = 'E resid KE')
ax1.set_xlabel('Ensemble size')
ax1.set_ylabel('Emission residual (% of prior)')
ax2.plot(NN,residuCKE,'b',label = 'C resid KE')
ax2.set_ylabel('Concentration residual (% of prior)')
ax1.legend(loc = 'lower right')
ax2.legend(loc = 'center right')
plt.savefig("VarN_KE"+add)
    
    




























