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
import geopandas as gpd
import pandas as pd

#%% Load field data

river1 = gpd.read_file(r'C:/Users/jmarggra/Documents/Measurements/GooseberryRiver/Terraces/Gooseberry_downstream_centerline.gpkg')
line1 = river1['geometry'][0]
river_DEM = pd.read_csv(r'\\wsl.localhost\Ubuntu\home\jessicamarggraf\data\GooseberryRiver\GooseberryRiver_05m\GooseberryRiver_05m_main_stem.csv')
outpath_figures = r'C:\Users\jmarggra\Documents\Modelling\Stream_power\Figures'
river_DEM['Distance_downstream'] = [river_DEM['flow_distance'].iloc[0] - river_DEM['flow_distance'][i] for i in range(len(river_DEM))]
river_DEM['Elevation_reference'] = [river_DEM['elevation'][i] -river_DEM['elevation'].iloc[-1] for i in range(len(river_DEM))]


#%% Define parameters and set up profile

# Model parameters
dt = 1 # time-step length
nodes = 100
river_length = 45459.5 #47000 # length Gooseberry River
dx = river_length/nodes 
x_range = np.linspace(0, river_length, nodes)
t_final = 13000

# Streampower law parameters
theta = 0.15 # Determined using chi-analysis
# m = 0.4
# n = m/theta
n = 1 # linear kinematic wave model
m = theta*n

K = 5E-6  #6.635e-6 in Rohrmann et al. 2023

# Basin parameters
# Varying slope S and area A for each node
A_max = river_DEM['drainage_area'].max()
A = np.linspace(0.1, A_max, 100) # relatively fix (m2) #220,000,000
backgroundU = 0.0045 # background uplift rate (m/year) (approx 5mm/yr based on literature)

#%% DEFINE PROFILE / instantiate model ##################################################
# How should the initial profile look like? Flat or an equilibrium profile?

# Define elevation
#zdot_channel_ini = 5E-3 # [m]
elevation_ini = 30
zz = elevation_ini * np.ones(x_range.shape)
theta_ini = 0.45 # Determined using chi-analysis
n_ini = 1 # linear kinematic wave model
m_ini = theta*n
K_ini = 3.5e-6 #6.635e-6 in Rohrmann et al. 2023
backgroundU_ini = 2E-4
nn = 1200000

time1 = 0
# Define variables
zz_ini = np.zeros((nn+1, nodes))
dz_b_ini = np.zeros((nn+1, nodes))
E_ini = np.zeros((nn+1, nodes))
while time1<nn:
    time1 += dt        
    # Calculate bedrock erosion using the stream power model    
    S = np.absolute(np.diff(zz)) # positive values are erosion
    S = np.append(S,S[-1])
    E = K_ini*A**m_ini*S**n_ini  
    dz_b = E            
    
    # ini variables
    dz_b_ini[int((time1/dt)),:] = -dz_b/dt
    zz_ini[int((time1/dt)),:] = zz-dz_b/dt
    E_ini[int((time1/dt)),:] = E/dt
        
    # Uplift for next round
    zz += backgroundU_ini*dt - dz_b
    zz[-1] = 0

# Plot initial profile
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)

for i in np.arange(0,nn,10000):
    ax.plot(x_range,zz_ini[i,:], lw = 1, color = 'lightgrey')   
ax.plot(x_range,zz, lw = 2, color = 'black')
   
ax.set_xlabel('Distance downstream [m]', fontsize=16, weight = 'bold')
ax.set_ylabel('Elevation [m]', fontsize=16, weight = 'bold')
ax.set_xlim(x_range[0],x_range[-1])   
ax.set_ylim(0,(np.max(zz)+0.1*np.max(zz)))
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax.text(0.05,0.4, ('K:',K_ini), transform = ax.transAxes) 
ax.text(0.05,0.35, ('U_ini:',backgroundU_ini), transform = ax.transAxes)
ax.text(0.05,0.3, ('Ele_ini:',elevation_ini), transform = ax.transAxes)
ax.text(0.05,0.25, ('nn:',nn), transform = ax.transAxes)

ax.plot(river_DEM['Distance_downstream'],river_DEM['Elevation_reference'], 
        ls = '-', lw = 1, color = 'navy')
fig.tight_layout()
figname = 'Longprofile_basic_streampower_ini'
fig.savefig(outpath_figures + '\\' + figname + str(K_ini) +'_' + str(backgroundU_ini) +'_'+str(elevation_ini)+'_'+str(nn)+'.png', dpi=300, bbox_inches='tight')

#%%## RUN MODEL ANALYSIS ####################################
time2 = 0
nn = t_final
zz_save = np.zeros((t_final+1, nodes))
dz_b_save = np.zeros((t_final+1, nodes))
E_save = np.zeros((t_final+1, nodes))
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
    

#%% Plot longprofile
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)

for i in np.arange(0,t_final,1000):
    ax.plot(x_range,zz_save[i,:], lw = 1, color = 'lightgrey')   
ax.plot(x_range,zz, lw = 2, color = 'black')

ax.plot(river_DEM['Distance_downstream'],river_DEM['Elevation_reference'], 
        ls = '-', lw = 1, color = 'navy')

# plt.plot(x_range,zz_save[5,:], lw = 1, color = 'red')    
ax.set_xlabel('Distance downstream [m]', fontsize=16, weight = 'bold')
ax.set_ylabel('Elevation [m]', fontsize=16, weight = 'bold')
ax.set_xlim(x_range[0],x_range[-1])   
# ax.set_xlim(40000,x_range[-1])   
ax.set_ylim(0,(np.max(zz)+0.1*np.max(zz)))
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)

fig.tight_layout()
figname = 'Longprofile_basic_streampower_with_Gooseberry'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi=300, bbox_inches='tight')


#%% Plot zoom of longprofile
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)

for i in np.arange(0,t_final,1000):
    ax.plot(x_range,zz_save[i,:], lw = 1, color = 'lightgrey')   
ax.plot(x_range,zz, lw = 2, color = 'black')

ax.plot(river_DEM['Distance_downstream'],river_DEM['Elevation_reference'], 
        ls = '-', lw = 1, color = 'navy')

# plt.plot(x_range,zz_save[5,:], lw = 1, color = 'red')    
ax.set_xlabel('Distance downstream [m]', fontsize=16, weight = 'bold')
ax.set_ylabel('Elevation [m]', fontsize=16, weight = 'bold')
# ax.set_xlim(x_range[0],x_range[-1])   
ax.set_xlim(40000,x_range[-1])   
ax.set_ylim(0,130)
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)

fig.tight_layout()
figname = 'Longprofile_basic_streampower_with_Gooseberry_zoom'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi=300, bbox_inches='tight')


#%% Plot bedrock incision
E_save_tot = E_save.sum(axis=0)
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)

ax.plot(x_range,E_save_tot, lw = 2, color = 'black')   
   
ax.set_xlabel('Distance downstream [m]', fontsize=16, weight = 'bold')
ax.set_ylabel('Bedrock incision [m]', fontsize=16, weight = 'bold')
ax.set_xlim(x_range[0],x_range[-1])  
ax.set_ylim(0,)
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)

fig.tight_layout()
figname = 'Bedrock_incision_basic_streampower_with_Gooseberry'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi=300, bbox_inches='tight')


#%% Export variables




















