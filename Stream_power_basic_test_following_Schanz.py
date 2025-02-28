# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:06:57 2025

@author: jmarggra
"""

# Load modules
import numpy as np
from matplotlib import pyplot as plt
import importlib
import pickle

#%% Define parameters and set up profile

# Model parameters
dt = 1 # time-step length
nodes = 100
river_length = 37000 # length Gooseberry River
dx = river_length/nodes 
x_range = np.linspace(0, river_length, nodes)
t_final = 13000

# Streampower law parameters
m = 0.4
theta = 0.45 # Determined using chi-analysis
n = m/theta
K = 1E-6 #3E-5

# Basin parameters
# Varying slope S and area A for each node
A = np.linspace(0.1, 220000000, 100) # relatively fix (m2)
backgroundU = 0.002 # background uplift rate (mm/year)

#%% DEFINE PROFILE / instantiate model ##################################################
# How should the initial profile look like? Flat or an equilibrium profile?

# Define elevation
zdot_channel_ini = 5E-3 # [m]
zz = 400 * np.ones(x_range.shape)

# Define variables
zz_save = np.zeros((t_final+1, nodes))
dz_b_save = np.zeros((t_final+1, nodes))
E_save = np.zeros((t_final+1, nodes))

# Plot initial profile
# plt.plot(x_range,zz)

### RUN MODEL ANALYSIS ####################################
time2 = 0
nn = t_final
while time2<nn:
    time2 += dt        
    # Calculate bedrock erosion using the stream power model    
    S = np.absolute(np.diff(zz)) # positive values are erosion
    S = np.append(S,S[-1])
    E = K*A**m*S**n  
    dz_b = E
    
    # save variables
    dz_b_save[int((time2/dt)),:] = -dz_b/dt
    zz_save[int((time2/dt)),:] = zz-dz_b/dt
    E_save[int((time2/dt)),:] = E/dt
        
    # Uplift for next round
    zz += backgroundU*dt - dz_b
    zz[-1] = 0
    
#%% Plot
for i in np.arange(0,t_final,1000):
    plt.plot(x_range,zz_save[i,:], lw = 1, color = 'lightgrey')   
plt.plot(x_range,zz, lw = 2, color = 'black')
# plt.plot(x_range,zz_save[5,:], lw = 1, color = 'red')    
plt.xlabel('Distance downstream [m]', fontsize=14)
plt.ylabel('Elevation [m]', fontsize=14)
plt.xlim(x_range[0],x_range[-1])   
plt.ylim(0,(np.max(zz)+0.1*np.max(zz)))
plt.show()


#%% Export variables




















