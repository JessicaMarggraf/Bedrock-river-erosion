# -*- coding: utf-8 -*-
"""
This script includes analysis and visualization of field- and experimentally 
determined data of rock and erosion properties. 

Related to article submitted to Earth Surface Processes and Landforms

Requires thin section and field data to be analysed and formatted 

Author: Jessica Marggraf (ORCID: 0000-0002-9372-5494)
Created: 2026-11-22
Repository: https://github.com/JessicaMarggraf/Bedrock-river-erosion
DOI: 10.5281/zenodo.1234567

"""
# Copyright (C) 2026 Jessica Marggraf
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#%% Import

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression

#%% Load data
# change filepath 
# field_data = pd.read_csv(r'filepath\All_field_data_article_outcrops.csv') 
field_data_lith_ave= pd.read_csv(r'filepath\Data_for_comparison_flume_meas.csv') 
data_thin_sections= pd.read_csv(r'filepath\Data_thin_sections_lithology_averaged.csv')
small_flume_data = pd.read_csv(r'filepath\Small_flume_results.csv')
large_flume_data = pd.read_csv(r'filepath\Large_flume_results.csv')
outpath_figures = r'filepath\Figures'

# data from the literature
data_Turowski = pd.read_csv(r'filepath\Turowski_data_table1.csv') 
data_literature_erosion_tensile = pd.read_csv(r'filepath\Literature_data_Fig7Turowski23.csv') 
data_energy = pd.read_csv(r'filepath\Kinetic_energy_data.csv') 
data_literature_erosion_tensile_add = pd.read_csv(r'filepath\Literature_data.csv') 
data_literature_Schmidt = pd.read_csv(r'filepath\Data_Schmidt_hammer.csv') 

# sent by Beer
data_Beer = pd.read_csv(r'filepath\Beer_concretedrops.txt', sep='\s+') 
data_Head = pd.read_csv(r'filepath\Head_glassdrops.txt', sep='\s+') 
data_Liu = pd.read_csv(r'filepath\Liu_concretemills.txt', sep='\s+') 
data_Scheingross = pd.read_csv(r'filepath\Scheingross_foammills.txt', sep='\s+') 
data_Sklar_bedrockdrops = pd.read_csv(r'filepath\Sklar_bedrockdrops.txt', sep='\s+') 
data_Sklar_bedrockmills = pd.read_csv(r'filepath\Sklar_bedrockmills.txt', sep='\s+') 

#%% Prepare general info
lithologies = ['Mns', 'Mis', 'Mnb_Bottom', 'Mng_Bottom', 'Bottom_flow_4', 'Mnb_Top', 'Mng_Top', 'Top_flow_3', 'Mni', 'Cst']
lithologies_label = ['Diab. Mls', 'Diab. Mli', 'Mnb C.', 'Mng C.', 'Mng #4 C.', 'Mnb Top', 'Mng Top', 'Mng #3 Top', 'Icelandite', 'Sandstone']
colors = ['peru', 'sienna', 'yellowgreen', 'dodgerblue', 'cornflowerblue', 'forestgreen', 'royalblue', 'blue', 'gold', 'violet']
markers = ['s', 's', 'v', 'v', 'v', '^', '^', '^', 'D', 'p']
sizes = [10,10,11,11,11,11,11,11,10,11]
abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Calculate tensile strength using Jamshidi et al., 2018
field_data_lith_ave['Tensile_strength_Pa'] = 0.0347*10**6*field_data_lith_ave['ISRM_Mean']**1.3783
field_data_lith_ave['Tensile_strength_Pa_std'] = 0.0347*10**6*field_data_lith_ave['ISRM_Std']**1.3783
field_data_lith_ave['Compressive_strength_Pa'] = 1.624*field_data_lith_ave['ISRM_Mean']- 17.23 # assuming vertical
field_data_lith_ave['Compressive_strength_Pa_std'] = (1.624*field_data_lith_ave['ISRM_Std']- 17.23)*(-1)

#%%############################################################################

# LAB DATA

#%%############################################################################
# Flume velocities and pebble diameters
# velocities given by the lab and diameters measured; both are min and max values
u_p_large_min = 1.3*0.7*1.3 # m/s
u_p_large_max = 1.5*0.7*1.3
u_p_small_min = 0.5*0.7*1.3
u_p_small_max = 0.55*0.7*1.3
D_p_small_min = 0.02 # m
D_p_small_max = 0.03
D_p_large_min = 0.06
D_p_large_max = 0.08

print('ratio1: ', 1.3/0.5*0.06/0.02)
print('ratio2: ', 1.5/0.55*0.08/0.03)
min_ratio_vel_diam =  1.5/0.55*0.08/0.03
max_ratio_vel_diam = 1.3/0.5*0.06/0.02

rho_s = 2650
# roughly estimated pebble mass assuming spheres (wrong)
m_small_min = 4/3*math.pi*(D_p_small_min/2)**3*rho_s #kg
m_small_max = 4/3*math.pi*(D_p_small_max/2)**3*rho_s
m_large_min = 4/3*math.pi*(D_p_large_min/2)**3*rho_s
m_large_max = 4/3*math.pi*(D_p_large_max/2)**3*rho_s

# Prepare data
data_thin_sections['Lithology'] = lithologies
# small flume: second run, values per pebble
small_flume_data = small_flume_data.sort_values(['Lithology_ID'])
small_flume_data = small_flume_data.reset_index(drop = True)
small_flume_data['Size_corr_macro_abrasion_rate_perc_h'] = (small_flume_data['Final_mass_fragment_g']/small_flume_data['Initial_mass_2nd_run_g'])/small_flume_data['Experiment_duration_h']
# adjust experiment duration with correct values


# large flume: 1st run: lithology-averaged values; 2nd (and 3rd) run: values per pebble
large_flume_data = large_flume_data.sort_values(['Lithology_ID'])
large_flume_data = large_flume_data.reset_index(drop = True)

# Calculate abraded volumes in large flume
large_flume_data['Volume_abraded_m3'] = (large_flume_data['Initial_mass_g']-large_flume_data['Final_mass_initial_size_class_g'])/large_flume_data['Density_g_cm3']/1e6
large_flume_data['Volume_micro_abraded_m3'] = (large_flume_data['Initial_mass_g']-large_flume_data['Final_total_mass_including_fragments_larger_1cm_g'])/large_flume_data['Density_g_cm3']/1e6
large_flume_data['Volume_macro_abraded_m3'] = (large_flume_data['Final_mass_fragment_g'])/large_flume_data['Density_g_cm3']/1e6

#%% Average flume data over number of pebbles to get lithology averages

# 1) Large flume data: use only the ones from the first run
large_flume_first_run = large_flume_data[large_flume_data["Run"]==1]
# average the lithology with two times in 1st run (ID 10) CFC
ab = large_flume_first_run[large_flume_first_run['Lithology_ID']==10].mean(numeric_only=True)
large_flume_first_run = large_flume_first_run.drop([21,23], errors='ignore')
large_flume_first_run = pd.concat([large_flume_first_run,pd.DataFrame(ab).transpose()], axis = 0)
large_flume_first_run = large_flume_first_run.reset_index(drop=True)
large_flume_first_run.iloc[9,2] = 'Cut Face Creek sandstone'
large_flume_first_run = large_flume_first_run.sort_values('Lithology_ID')
large_flume_first_run = large_flume_first_run.reset_index(drop=True)
large_flume_first_run['Mass_loss_g'] = large_flume_first_run['Initial_mass_g']-large_flume_first_run['Final_total_mass_including_fragments_larger_1cm_g']
large_flume_first_run['Label'] = lithologies_label

# 2) Small flume data
# data are available per pebble
aa = []
ad = []
for i in np.arange(1,11,1):
    ab = small_flume_data[small_flume_data['Lithology_ID']==i].mean(numeric_only=True)
    aa.append(ab)
    ac = small_flume_data[small_flume_data['Lithology_ID']==i].std(numeric_only=True)
    ad.append(ac)
small_flume_data_average = pd.DataFrame(aa)
small_flume_data_average_std = pd.DataFrame(ad)
small_flume_data_average_std = small_flume_data_average_std.iloc[:,2:]
small_flume_data_average_std.columns = [col + '_std' for col in small_flume_data_average_std.columns]
small_flume_data_average = pd.concat([small_flume_data_average,small_flume_data_average_std], axis = 1)
small_flume_data_average = small_flume_data_average.sort_values('Lithology_ID')
small_flume_data_average['Fragmented_pebbles_perc'] = [0,0,0,0,0,25,20,40,0,0] # calculated from data
small_flume_data_average['Initial_no'] = [5,5,4,5,6,4,11,5,5,4]
# small_flume_data_average['Mass_loss_g'] = small_flume_data_average['Initial_mass_g']-small_flume_data_average[]

#%% Large flume second run data
# data are available per pebble, but only for a few lithologies
# Get data
large_flume_sec_run = large_flume_data[large_flume_data["Run"]>1]
large_flume_sec_run = large_flume_sec_run.dropna(subset =['Size_corr_attrition_rate_perc_h'])
large_flume_sec_run = large_flume_sec_run.reset_index(drop = True)
large_flume_sec_run_lithologies = large_flume_sec_run['Lithology_ID'].to_list()
large_flume_first_run['Size_corr_macro_abrasion_subtraction_rate_perc_h'] = large_flume_first_run['Size_corr_abrasion_rate_perc_h']-large_flume_first_run['Size_corr_attrition_rate_perc_h']
large_flume_sec_run['Size_corr_macro_abrasion_subtraction_rate_perc_h'] = large_flume_sec_run['Size_corr_abrasion_rate_perc_h']-large_flume_sec_run['Size_corr_attrition_rate_perc_h']
large_flume_first_run['Size_corr_macro_abrasion_rate_perc_h'] = (large_flume_first_run['Final_mass_fragment_g']/large_flume_first_run['Initial_mass_g'])/large_flume_first_run['Experiment_duration_h']
large_flume_sec_run['Size_corr_macro_abrasion_rate_perc_h'] = (large_flume_sec_run['Final_mass_fragment_g']/large_flume_sec_run['Initial_mass_g'])/large_flume_sec_run['Experiment_duration_h']

# Average second run data
aa = []
ad = []
for i in np.arange(1,11,1):
    ab = large_flume_sec_run[large_flume_sec_run['Lithology_ID']==i].mean(numeric_only=True)
    aa.append(ab)
    ac = large_flume_sec_run[large_flume_sec_run['Lithology_ID']==i].std(numeric_only=True)
    ad.append(ac)
large_flume_sec_run_average = pd.DataFrame(aa)
large_flume_sec_run_average_std = pd.DataFrame(ad)
large_flume_sec_run_average_std = large_flume_sec_run_average_std.iloc[:,2:]
large_flume_sec_run_average_std.columns = [col + '_std' for col in large_flume_sec_run_average_std.columns]
large_flume_sec_run_average = pd.concat([large_flume_sec_run_average,large_flume_sec_run_average_std], axis = 1)
large_flume_sec_run_average = large_flume_sec_run_average.sort_values('Lithology_ID')
large_flume_sec_run_average = large_flume_sec_run_average.dropna(subset =['ID'])
large_flume_sec_run_average = large_flume_sec_run_average.reset_index(drop = True)

#%% Combine data 
# Concat averaged large flume results (1st and 2nd run)
large_flume_first_run1 = large_flume_first_run.copy(deep=True)
large_flume_first_run1.columns = [
    col + "_1st_run" if i >= 4 else col
    for i, col in enumerate(large_flume_first_run1.columns)]

large_flume_sec_run_average1 = large_flume_sec_run_average.copy(deep=True)
large_flume_sec_run_average1.columns = [
    col + "_2nd_run" if i >= 4 else col
    for i, col in enumerate(large_flume_sec_run_average1.columns)]
key = 'Lithology_ID'
aligned_short = (
    large_flume_sec_run_average1.set_index(key).reindex(large_flume_first_run1[key]).reset_index(drop=True))
large_flume_average = pd.concat([large_flume_first_run1.reset_index(drop=True), aligned_short], axis=1) ##
 
# Combine averaged flume data
small_flume_data_average1 = small_flume_data_average.copy(deep=True)
small_flume_data_average1.columns = [col + '_small_flume' for col in small_flume_data_average1.columns]
large_flume_average1 = large_flume_average.copy(deep=True)
large_flume_average1.columns = [col + '_large_flume' for col in large_flume_average1.columns]

# Combine averaged flume data
flume_data = pd.concat([small_flume_data_average1,large_flume_average1], axis = 1)

# Combine flume and field data - averaged
data_averaged = pd.concat([field_data_lith_ave,flume_data], axis = 1)

#%%  
# calculate macro-abrasion rate in small flume
data_averaged['Fragmented_pebbles_perc_small_flume'] = [0,0,0,0,0,25,20,40,0,0]
D_p_small_mean = (D_p_small_min + D_p_small_max)/2
m_small_mean = (m_small_min + m_small_max)/2

# lithologies_large_single = large_flume_sec_run['Lithology_ID'].unique()
# lithologies_large_single = [lithologies_label[large_flume_sec_run.iloc[i]-1] for i in range(len(large_flume_sec_run))]
data_averaged['Size_corr_attrition_ratio_1st_large_small'] = data_averaged['Size_corr_attrition_rate_perc_h_1st_run_large_flume']/data_averaged['Size_corr_attrition_rate_perc_h_small_flume']
data_averaged['Size_corr_attrition_ratio_2nd_large_small'] = data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume']/data_averaged['Size_corr_attrition_rate_perc_h_small_flume']
mean_size_corr_attrition_ratio_1st_large_small = data_averaged['Size_corr_attrition_ratio_1st_large_small'].mean()
mean_size_corr_attrition_ratio_2nd_large_small = data_averaged[ "Size_corr_attrition_ratio_2nd_large_small"].mean()
# data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'] = data_averaged['Size_corr_abrasion_rate_perc_h_1st_run_large_flume']-data_averaged['Size_corr_attrition_rate_perc_h_1st_run_large_flume']
# data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'] = data_averaged['Size_corr_abrasion_rate_perc_h_2nd_run_large_flume']-data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume']

# compare abrasion rate in the large flume with attrition rate in the small flume - not sure if this is a good idea
data_averaged['Size_corr_abrasion_ratio_1st_large_small'] = data_averaged['Size_corr_abrasion_rate_perc_h_1st_run_large_flume']/data_averaged['Size_corr_attrition_rate_perc_h_small_flume']
data_averaged['Size_corr_abrasion_ratio_2nd_large_small'] = data_averaged['Size_corr_abrasion_rate_perc_h_2nd_run_large_flume']/data_averaged['Size_corr_attrition_rate_perc_h_small_flume']
mean_size_corr_abrasion_ratio_1st_large_small = data_averaged['Size_corr_abrasion_ratio_1st_large_small'].mean()
mean_size_corr_abrasion_ratio_2nd_large_small = data_averaged[ "Size_corr_abrasion_ratio_2nd_large_small"].mean()

# Calculate rates in g/h
data_averaged['Size_corr_attrition_rate_g_h_1st_run_large_flume'] = ((data_averaged['Initial_mass_g_1st_run_large_flume']+ data_averaged['Final_total_mass_including_fragments_larger_1cm_g_1st_run_large_flume'])/2)*data_averaged['Size_corr_attrition_rate_perc_h_1st_run_large_flume']
data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'] = ((data_averaged['Initial_mass_g_2nd_run_large_flume']+ data_averaged['Final_total_mass_including_fragments_larger_1cm_g_2nd_run_large_flume'])/2)*data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume']
# data_averaged['Size_corr_attrition_rate_g_h_1st_run_large_flume'] = (data_averaged['Initial_mass_g_1st_run_large_flume'])*data_averaged['Size_corr_attrition_rate_perc_h_1st_run_large_flume']
# data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'] = (data_averaged['Initial_mass_g_2nd_run_large_flume'])*data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume']
data_averaged['Size_corr_attrition_rate_g_h_small_flume'] = (data_averaged['Initial_mass_g_small_flume'])*data_averaged['Size_corr_attrition_rate_perc_h_small_flume']
data_averaged['Size_corr_attrition_ratio_1st_large_small_g_h'] = data_averaged['Size_corr_attrition_rate_g_h_1st_run_large_flume']/data_averaged['Size_corr_attrition_rate_g_h_small_flume']
data_averaged['Size_corr_attrition_ratio_2nd_large_small_g_h'] = data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume']/data_averaged['Size_corr_attrition_rate_g_h_small_flume']
mean_size_corr_attrition_ratio_1st_large_small_g_h = data_averaged['Size_corr_attrition_ratio_1st_large_small_g_h'].mean()
mean_size_corr_attrition_ratio_2nd_large_small_g_h = data_averaged[ "Size_corr_attrition_ratio_2nd_large_small_g_h"].mean()

data_averaged['Size_corr_abrasion_rate_g_h_1st_run_large_flume'] = ((data_averaged['Initial_mass_g_1st_run_large_flume']+ data_averaged['Final_total_mass_including_fragments_larger_1cm_g_1st_run_large_flume'])/2)*data_averaged['Size_corr_abrasion_rate_perc_h_1st_run_large_flume'] # use this conversion for large masses
data_averaged['Size_corr_abrasion_rate_g_h_2nd_run_large_flume'] = ((data_averaged['Initial_mass_g_2nd_run_large_flume']+ data_averaged['Final_total_mass_including_fragments_larger_1cm_g_2nd_run_large_flume'])/2)*data_averaged['Size_corr_abrasion_rate_perc_h_2nd_run_large_flume']
data_averaged['Size_corr_abrasion_rate_g_h_small_flume'] = (data_averaged['Initial_mass_g_small_flume'])*data_averaged['Size_corr_attrition_rate_perc_h_small_flume']
data_averaged['Size_corr_abrasion_ratio_1st_large_small_g_h'] = data_averaged['Size_corr_abrasion_rate_g_h_1st_run_large_flume']/data_averaged['Size_corr_abrasion_rate_g_h_small_flume']
data_averaged['Size_corr_abrasion_ratio_2nd_large_small_g_h'] = data_averaged['Size_corr_abrasion_rate_g_h_2nd_run_large_flume']/data_averaged['Size_corr_abrasion_rate_g_h_small_flume']
mean_size_corr_abrasion_ratio_1st_large_small_g_h = data_averaged['Size_corr_abrasion_ratio_1st_large_small_g_h'].mean()
mean_size_corr_abrasion_ratio_2nd_large_small_g_h = data_averaged[ "Size_corr_abrasion_ratio_2nd_large_small_g_h"].mean()

data_averaged['Size_corr_macro_abrasion_rate_g_h_1st_run_large_flume'] = ((data_averaged['Initial_mass_g_1st_run_large_flume']+ data_averaged['Final_total_mass_including_fragments_larger_1cm_g_1st_run_large_flume'])/2)*data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'] # use this conversion for large masses
data_averaged['Size_corr_macro_abrasion_rate_g_h_2nd_run_large_flume'] = ((data_averaged['Initial_mass_g_2nd_run_large_flume']+ data_averaged['Final_total_mass_including_fragments_larger_1cm_g_2nd_run_large_flume'])/2)*data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume']
data_averaged['Size_corr_macro_abrasion_rate_g_h_small_flume'] = (data_averaged['Initial_mass_g_small_flume'])*data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume']

#%% Plot small flume data
fig, ax = plt.subplots(10, 1, figsize=(10, 10), dpi=200)

for i in range(len(lithologies_label)):
    aa = small_flume_data[small_flume_data['Lithology_ID']==i+1]
    ab = aa.iloc[:,7]*100
    ax[i].hist(ab, bins= np.logspace(np.log10(0.001), np.log10(500),50), edgecolor='black', color = colors[i]) #np.logspace(np.log10(min(ab)), np.log10(max(ab+1)), len(ab))
    ax[i].set_xlim(0.1,12) 
    ax[i].set_ylim(0, 7) 
    ax[i].set_xscale('log') 
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].text(0.11, 5, str(abc[i] + ') ' + lithologies_label[i]), fontsize = 14)  
    # ax[i].set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    # ax[i].set_yticklabels(['0', '', '', '3', '', '', '6', ''])  
    ax[i].tick_params(axis = 'y', which = 'major', labelsize = 14) 
    if i < 9:
        ax[i].set_xticklabels([])
    else:
        ax[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax[i].tick_params(axis = 'x', which = 'major', labelsize = 14) 
        ax[i].ticklabel_format(style='plain', axis='x')
        
fig.supxlabel('Attrition rate (%/h)', fontsize=18, weight = 'bold') 
fig.supylabel('Number of pebbles (-)', fontsize=18, weight = 'bold') 
fig.tight_layout()
figname = 'Small_flume_results'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot small flume data macro-abrasion
fig, ax = plt.subplots(10, 1, figsize=(5, 10), dpi=200)

for i in range(len(lithologies_label)):
    aa = small_flume_data[small_flume_data['Lithology_ID']==i+1]
    ab = aa.iloc[:,-1]*100
    ax[i].hist(ab, bins= np.logspace(np.log10(0.001), np.log10(500),50), edgecolor='black', color = colors[i]) #np.logspace(np.log10(min(ab)), np.log10(max(ab+1)), len(ab))
    ax[i].set_xlim(3,100) 
    ax[i].set_ylim(0, 1.5) 
    ax[i].set_xscale('log') 
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].text(3.5, 1.3, str(abc[i] + ') ' + lithologies_label[i]), fontsize = 14)  
    # ax[i].set_yticks([0, 1, 2,])
    # ax[i].set_yticklabels(['0', '', '', '3'])  
    ax[i].tick_params(axis = 'y', which = 'major', labelsize = 14) 
    if i < 9:
        ax[i].set_xticklabels([])
    else:
        ax[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax[i].tick_params(axis = 'x', which = 'major', labelsize = 14) 
        ax[i].ticklabel_format(style='plain', axis='x')
        
fig.supxlabel('Macro-abrasion rate (%/h)', fontsize=18, weight = 'bold') 
fig.supylabel('Number of pebbles (-)', fontsize=18, weight = 'bold') 
fig.tight_layout()
figname = 'Small_flume_results_macro_abrasion'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot small flume data- fragmentation
fig, ax = plt.subplots(10, 1, figsize=(4, 10), dpi=200)

for i in range(len(lithologies_label)):
    ab = small_flume_data_average['Fragmented_pebbles_perc'].iloc[i]
    ax[i].bar(lithologies_label[i], ab, edgecolor='black', color = colors[i]) 
    ax[i].set_xlim(-1,1) 
    ax[i].set_ylim(0,50) 
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].text(-0.95, 45, str(abc[i] + ') ' + lithologies_label[i]), fontsize = 14)  
    # ax[i].set_yticks([0, 1, 2, 3])
    ax[i].set_xticklabels([])  
    ax[i].tick_params(axis = 'y', which = 'major', labelsize = 14) 
       
fig.supylabel('Fragmented pebbles (%)', fontsize=18, weight = 'bold') 
fig.supxlabel('Lithology', fontsize=18, weight = 'bold') 
fig.tight_layout()
figname = 'Small_flume_results_fragmentation'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')


#%% Plot small flume data- macro-abrasion
fig, ax = plt.subplots(10, 1, figsize=(4, 10), dpi=200)

for i in range(len(lithologies_label)):
    ab = small_flume_data_average['Size_corr_macro_abrasion_rate_perc_h'].iloc[i]*100
    ax[i].bar(lithologies_label[i], ab, edgecolor='black', color = colors[i]) 
    ax[i].set_xlim(-1,1) 
    ax[i].set_ylim(0,20) 
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].text(-0.95, 17, str(abc[i] + ') ' + lithologies_label[i]), fontsize = 14)  
    # ax[i].set_yticks([0, 1, 2, 3])
    ax[i].set_xticklabels([])  
    ax[i].tick_params(axis = 'y', which = 'major', labelsize = 14) 
       
fig.supylabel('Macro-abrasion rate (%/h)', fontsize=18, weight = 'bold') 
fig.supxlabel('Lithology', fontsize=18, weight = 'bold') 
fig.tight_layout()
figname = 'Small_flume_results_macro_abrasion'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot large flume data 3
fig, ax = plt.subplots(2, 3, figsize=(16,12), dpi=200)

# abrasion
for i in range(len(large_flume_first_run)):
    ax[0,0].plot(large_flume_first_run['Label'][i], large_flume_first_run['Size_corr_abrasion_rate_perc_h'][i]*100,
            marker = markers[i], markeredgecolor = colors[i], markersize = sizes[i]+2, color = 'None', ls = '', markeredgewidth = 2)  
    
ax[0,0].text(0.02, 0.95, 'a)', fontsize = 20, transform = ax[0,0].transAxes)            
ax[0,0].set_ylabel('Abrasion rate (%/h)', fontsize = 25, weight = 'bold')
ax[0,0].tick_params(axis = 'both', which = 'major', labelsize = 20)
ax[0,0].set_ylim(1,400) 
ax[0,0].set_yscale('log')
ax[0,0].set_xticklabels(lithologies_label, rotation = 90 )

# macro-abrasion
for i in range(len(large_flume_first_run)):
    ax[0,1].plot(large_flume_first_run['Label'][i], large_flume_first_run['Size_corr_macro_abrasion_rate_perc_h'][i]*100,
            marker = markers[i], markeredgecolor = colors[i], markersize = sizes[i]+2, color = 'None', ls = '', markeredgewidth = 2)  
    
ax[0,1].text(0.02, 0.95, 'b)', fontsize = 20, transform = ax[0,1].transAxes)  
# ax[0,1].set_ylabel('Macro-abrasion rate (%/h)', fontsize = 25, weight = 'bold')
ax[0,1].set_ylabel('Macro-abrasion rate (%/h)', fontsize = 25, weight = 'bold')
ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = 20)
ax[0,1].set_ylim(1,400) 
ax[0,1].set_yscale('log')
ax[0,1].set_xticklabels(lithologies_label, rotation = 90 )

# attrition
ax[0,2].plot(1, large_flume_first_run['Size_corr_attrition_rate_perc_h'][1]*100,
        marker = markers[0], color = 'None', markersize = sizes[1]+2, ls = '', 
        markeredgecolor = 'black', markeredgewidth = 2, label = r'$1^{\rm{st}}$ run', zorder =0)

for i in range(len(large_flume_first_run)):
    ax[0,2].plot(large_flume_first_run['Label'][i], large_flume_first_run['Size_corr_attrition_rate_perc_h'][i]*100,
            marker = markers[i], markeredgecolor = colors[i], markersize = sizes[i]+2, color = 'None', ls = '', markeredgewidth = 2, zorder = 50)  
    
ax[0,2].legend(loc = 'lower right', fontsize = 20)
ax[0,2].text(0.02, 0.95, 'c)', fontsize = 20, transform = ax[0,2].transAxes)  
# ax[0,2].set_ylabel('Attrition rate (%/h)', fontsize = 25, weight = 'bold')
ax[0,2].set_ylabel('Attrition rate (%/h)', fontsize = 25, weight = 'bold')
ax[0,2].tick_params(axis = 'both', which = 'major', labelsize = 20)
ax[0,2].set_ylim(1,400) 
ax[0,2].set_yscale('log')
ax[0,2].set_xticklabels(lithologies_label, rotation = 90 )

# abrasion
for i in range(len(large_flume_sec_run)):
    aa = large_flume_sec_run['Lithology_ID'].iloc[i]
    ax[1,0].plot(aa-1, large_flume_sec_run['Size_corr_abrasion_rate_perc_h'][i]*100,
            marker = markers[aa-1], color = colors[aa-1], markersize = sizes[aa-1]+2, ls = '', markeredgecolor = 'black', markeredgewidth = 0.2)
    
ax[1,0].text(0.02, 0.95, 'd)', fontsize = 20, transform = ax[1,0].transAxes)            
ax[1,0].set_ylabel('Abrasion rate (%/h)', fontsize = 25, weight = 'bold')
ax[1,0].tick_params(axis = 'both', which = 'major', labelsize = 20)
ax[1,0].set_ylim(1,400) 
ax[1,0].set_yscale('log')
ax[1,0].set_xticklabels(lithologies_label, rotation = 90 )

# macro-abrasion
for i in range(len(large_flume_sec_run)):
    aa = large_flume_sec_run['Lithology_ID'].iloc[i]
    ax[1,1].plot(aa-1, large_flume_sec_run['Size_corr_macro_abrasion_rate_perc_h'][i]*100,
            marker = markers[aa-1], color = colors[aa-1], markersize = sizes[aa-1]+2, ls = '', markeredgecolor = 'black', markeredgewidth = 0.2)

ax[1,1].text(0.02, 0.95, 'e)', fontsize = 20, transform = ax[1,1].transAxes)  
# ax[1,1].set_ylabel('Macro-abrasion rate (%/h)', fontsize = 25, weight = 'bold')
ax[1,1].set_ylabel('Macro-abrasion rate (%/h)', fontsize = 25, weight = 'bold')
ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = 20)
ax[1,1].set_ylim(1,400) 
ax[1,1].set_yscale('log')
ax[1,1].set_xticklabels(lithologies_label, rotation = 90 )

# attrition
ax[1,2].plot(0, large_flume_sec_run['Size_corr_attrition_rate_perc_h'][0]*100,
        marker = markers[0], color = 'black', markersize = sizes[0], ls = '', 
        markeredgecolor = 'black', markeredgewidth = 0.2, label = r'$2^{\rm{nd}}$ run')

for i in range(len(large_flume_sec_run)):
    aa = large_flume_sec_run['Lithology_ID'].iloc[i]
    ax[1,2].plot(aa-1, large_flume_sec_run['Size_corr_attrition_rate_perc_h'][i]*100,
            marker = markers[aa-1], color = colors[aa-1], markersize = sizes[aa-1]+2, ls = '', markeredgecolor = 'black', markeredgewidth = 0.2)

ax[1,2].legend(loc = 'lower right', fontsize = 20)
ax[1,2].text(0.02, 0.95, 'f)', fontsize = 20, transform = ax[1,2].transAxes)  
# ax[1,2].set_ylabel('Attrition rate (%/h)', fontsize = 25, weight = 'bold')
ax[1,2].set_ylabel('Attrition rate (%/h)', fontsize = 25, weight = 'bold')
ax[1,2].tick_params(axis = 'both', which = 'major', labelsize = 20)
ax[1,2].set_ylim(1,400) 
ax[1,2].set_yscale('log')
ax[1,2].set_xticklabels(lithologies_label, rotation = 90 )

fig.tight_layout()
figname = 'Large_flume_results_log_2rows'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

ratio_large_sec_run = large_flume_sec_run['Size_corr_macro_abrasion_rate_perc_h']/large_flume_sec_run['Size_corr_attrition_rate_perc_h']
ratio_large_first_run = large_flume_first_run['Size_corr_macro_abrasion_rate_perc_h']/large_flume_first_run['Size_corr_attrition_rate_perc_h']

#%%############################################################################

# DISCUSSION

#%%############################################################################


#%%############################################################################
# Analyse flume data

#%% Ratios
# Kinetic energy
KE_small_min = 1/2*m_small_min*u_p_small_min**2 # J
KE_small_max = 1/2*m_small_max*u_p_small_max**2
KE_large_min = 1/2*m_large_min*u_p_large_min**2
KE_large_max = 1/2*m_large_max*u_p_large_max**2
KE_large_mean = (KE_large_min+KE_large_max)/2
KE_small_mean = (KE_small_min+KE_small_max)/2

# divide KE by impactor
KE_D_small_min = KE_small_min/D_p_small_min**2
KE_D_small_max = KE_small_max/D_p_small_max**2
KE_D_large_min = KE_large_min/D_p_large_min**2
KE_D_large_max = KE_large_max/D_p_large_max**2

print(KE_large_max/KE_small_max)
print(KE_large_min/KE_small_min)


#%% Ratios test with 2up
# Flume velocities and pebble diameters
u_p_large_min2 = u_p_large_min*2 # m/s
u_p_large_max2 = u_p_large_max*2
u_p_small_min2 = u_p_small_min*2
u_p_small_max2 = u_p_small_max*2

# Kinetic energy
KE_small_min2 = 1/2*m_small_min*u_p_small_min2**2 # J
KE_small_max2 = 1/2*m_small_max*u_p_small_max2**2
KE_large_min2 = 1/2*m_large_min*u_p_large_min2**2
KE_large_max2 = 1/2*m_large_max*u_p_large_max2**2

# divide KE by impactor
KE_D_small_min2 = KE_small_min2/D_p_small_min**2
KE_D_small_max2 = KE_small_max2/D_p_small_max**2
KE_D_large_min2 = KE_large_min2/D_p_large_min**2
KE_D_large_max2 = KE_large_max2/D_p_large_max**2

#%% Calculate Stokes number
u_w_small = 1.15
u_w_large = 2.4
rho_w = 1000
nu = 0.001 # Pa s - dynamic viscosity of water
mu = 1e-6

# standard
St_small_min = (rho_s*u_p_small_min*D_p_small_min)/(9*nu)
St_small_max = (rho_s*u_p_small_max*D_p_small_max)/(9*nu)
St_large_min = (rho_s*u_p_large_min*D_p_large_min)/(9*nu)
St_large_max = (rho_s*u_p_large_max*D_p_large_max)/(9*nu)

# Reynolds number
Re_small_min = ((u_w_small-u_p_small_min)*D_p_small_min)/mu
Re_small_max = ((u_w_small-u_p_small_max)*D_p_small_max)/mu
Re_large_min = ((u_w_large-u_p_large_min)*D_p_large_min)/mu
Re_large_max = ((u_w_large-u_p_large_max)*D_p_large_max)/mu

# collisional Stokes number
St_col_small_min = (rho_s*D_p_small_min*u_p_small_min)/(18*nu)
St_col_small_max = (rho_s*D_p_small_max*u_p_small_max)/(18*nu)
St_col_large_min = (rho_s*u_p_large_min*D_p_large_min)/(18*nu)
St_col_large_max = (rho_s*D_p_large_max*u_p_large_max)/(18*nu)

#%% Plot attrition ratio small - large flume - 2

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=200)    
# use second run small flume and 2nd run large flume
for i in range(len(data_averaged)):
    ax.plot(large_flume_first_run['Label'][i], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume'][i]/data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][i],
            marker = markers[i], color = colors[i], markersize = sizes[i]+2, ls = '', markeredgecolor = 'black', 
            markeredgewidth = 0.2)     

ax.plot(0,  data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume'][0]/data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][0],
        marker = markers[0], color = 'black', markersize = sizes[0], ls = '', zorder = 0,
        markeredgecolor = 'black', markeredgewidth = 0.2, label = r'$2^{\rm{nd}}$ run')

# ax.axhspan(min_ratio_vel_diam,max_ratio_vel_diam, -0.5, 10.5, color ='lightgrey', label = r'$r_{\rm{u,D}}$')
ax.hlines(mean_size_corr_attrition_ratio_2nd_large_small, -0.5, 10.5, lw =1, color = 'black', zorder =0, label = r'${\overline{r_{att,l,s}}}$')
# ax.hlines(6, -0.5, 10.5, lw =1, ls = "--", color = 'black', zorder =0, label = r'$r_{\rm{att,Lavé}}$')

ax.legend(fontsize = 18)
ax.set_ylabel('Attrition rate large/small flume (-)', fontsize=18, weight = 'bold')
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax.set_xticklabels(lithologies_label, rotation = 90 )
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(0,13) 

fig.tight_layout()
figname = 'Large_small_flume_attrition_ratio_2nd'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')


#%% Plot macro-abrasion attrition ratio small - large flume
ratio_size_corr_macro_abrasion_ratio_large_small = [data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume']/(data_averaged['Fragmented_pebbles_perc_small_flume']/100),
                                                   data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume']/(data_averaged['Fragmented_pebbles_perc_small_flume']/100)]
ratio_size_corr_macro_abrasion_ratio_large_small = pd.concat(ratio_size_corr_macro_abrasion_ratio_large_small)
ratio_size_corr_macro_abrasion_ratio_large_small.replace([np.inf, -np.inf], 0, inplace=True)
mean_size_corr_macro_abrasion_ratio_large_small = ratio_size_corr_macro_abrasion_ratio_large_small.mean()

fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=200)    

# Attrition
for i in range(len(data_averaged)):
    ax[0].plot(large_flume_first_run['Label'][i], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume'][i]/data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][i],
            marker = markers[i], color = colors[i], markersize = sizes[i]+2, ls = '', markeredgecolor = 'black', 
            markeredgewidth = 0.2)     

# ax[0].axhspan(min_ratio_vel_diam,max_ratio_vel_diam, -0.5, 10.5, color ='lightgrey', label = r'$r_{\rm{u,D}}$')
ax[0].hlines(mean_size_corr_attrition_ratio_2nd_large_small, -0.5, 10.5, lw =1, color = 'black', zorder =0, label = r'${\overline{r_{micro,l/s}}}$')
# ax[0].hlines(6, -0.5, 10.5, lw =1, ls = "--", color = 'black', zorder =0, label = r'$r_{\rm{att,Lavé}}$')

ax[0].plot(0,  data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume'][0]/data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][0],
        marker = markers[0], color = 'black', markersize = sizes[0], ls = '', zorder = 0,
        markeredgecolor = 'black', markeredgewidth = 0.2, label = r'$2^{\rm{nd}}$ run')

ax[0].legend(fontsize = 18)
ax[0].set_ylabel('Attrition rate large/small flume (-)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis = 'both', which = 'major', labelsize = 14)
ax[0].set_xticklabels(lithologies_label, rotation = 90 )
ax[0].set_xlim(-0.5, 9.5)
ax[0].set_ylim(-0.2,13.5) 

# Macro-abrasion
for i in range(len(data_averaged)):
    aa = data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]/(data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i]/100)
    # if aa==np.inf:
    #     aa = 0
    ax[1].plot(large_flume_first_run['Label'][i], aa,
            marker = markers[i], color = 'None', markersize = sizes[i]+2, ls = '', markeredgecolor = colors[i], 
            markeredgewidth = 2)     
for i in range(len(data_averaged)):
    aa = data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]/(data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i]/100)
    # if aa==np.inf:
    #     aa = 0
    ax[1].plot(large_flume_first_run['Label'][i], aa,
            marker = markers[i], color = colors[i], markersize = sizes[i]+2, ls = '', markeredgecolor = 'black', 
            markeredgewidth = 0.2)  
    
# ax[1].axhspan(min_ratio_vel_diam,max_ratio_vel_diam, -0.5, 10.5, color ='lightgrey', label = r'$r_{\rm{u,D}}$')
ax[1].hlines(mean_size_corr_macro_abrasion_ratio_large_small, -0.5, 10.5, lw =1, color = 'black', zorder = 50, label = r'${\overline{r_{macro,l/s}}}$')
# ax[1].hlines(6, -0.5, 10.5, lw =1, ls = "--", color = 'black', zorder =0, label = r'$r_{\rm{att,Lavé}}$')

ax[1].plot(1, data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]/((data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i])/100),
        marker = markers[0], color = 'None', markersize = sizes[1]+2, ls = '', 
        markeredgecolor = 'black', markeredgewidth = 2, label = r'$1^{\rm{st}}$ run', zorder =0)
ax[1].plot(0, data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]/(data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i]/100),
        marker = markers[0], color = 'black', markersize = sizes[0], ls = '', 
        markeredgecolor = 'black', markeredgewidth = 0.2, label = r'$2^{\rm{nd}}$ run')

ax[1].legend(frameon=True, framealpha = 1, facecolor  ='white', fontsize = 18)
ax[1].set_ylabel('Macro-abrasion rate large/small flume (-)', fontsize=18, weight = 'bold')
ax[1].tick_params(axis = 'both', which = 'major', labelsize = 14)
ax[1].set_xticklabels(lithologies_label, rotation = 90 )
ax[1].set_xlim(-0.5, 9.5)
# ax[1].set_ylim(-0.2, 13.5) 

ax[0].text(0.02, 0.95, 'a)', fontsize = 14, transform = ax[0].transAxes) 
ax[1].text(0.02, 0.95, 'b)', fontsize = 14, transform = ax[1].transAxes) 

fig.tight_layout()
figname = 'Large_small_flume_macroabrasion_attrition_ratio'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')

#%% Prepare Fragment diameter - mean kinetic impact energy density

fragment_d = 0.004 #m
attrition_d = 0.000
ke_d_small_xrange = np.linspace(KE_D_small_min,KE_D_small_max, 10)
ke_d_large_xrange = np.linspace(KE_D_large_min,KE_D_large_max, 10)
ke_d_large_xrange2 = np.linspace(KE_D_large_min,KE_D_large_max, 3)
ke_d_small_xrange2 = np.linspace(KE_D_small_min,KE_D_small_max, 3)
aa = data_averaged[data_averaged['Fragmented_pebbles_perc_small_flume']>0]
ab = data_averaged[data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume']>0]


#%% Plot Fragment diameter - mean kinetic impact energy density
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

ax.axvspan(KE_D_small_min, 3400, ymin= 0, ymax= 12, color = 'peachpuff', alpha = 0.5, lw = 0.5, ec = 'black', zorder = 0) # transition zone

# Beer & Lamb (2021)
ax.hlines(0.002, 0.00001,100000,  color = 'black', lw = 1, ls = '--')
ax.axvspan(1900, 3400, ymin= 0, ymax= 12, color = 'tan', lw = 0.5, ec = 'black') # transition zone
data_Beer['Mean_volume_fragment'] = data_Beer["total.fragV"]/data_Beer['n_frag'] # assuming a sphere
data_Beer['Diameter_fragment'] = ((6*data_Beer["Mean_volume_fragment"])/(math.pi))**(1/3)
 
# other data literature
y_Head = len(data_Head)*[0.00005]
for i in range(len(y_Head)):
    ax.plot(data_Head['Eeff'][i]/((y_Head[i])**2), attrition_d, marker = "H", color = "lightblue", markersize = 10, ls = '', markeredgecolor = 'black', 
        markeredgewidth = 0.2)
ax.plot(data_Head['Eeff'][0]/((y_Head[0])**2), attrition_d, marker = "H", color = "lightblue", markersize = 10, ls = '', markeredgecolor = 'black', 
        markeredgewidth = 0.2, zorder = 0, label = 'Head & Harr, (1970), glass')
y_Liu = len(data_Liu)*[0.017]  # grain size should be ok, although not precise
for i in range(len(y_Liu)):
    ax.plot(data_Liu['mean.E_kin'][i]/((y_Liu[i])**2), attrition_d, marker = 'v', color = "grey", markersize = 10, ls = '', markeredgecolor = 'black', 
        markeredgewidth = 0.2)
ax.plot(data_Liu['mean.E_kin'][0]/((y_Liu[0])**2), attrition_d, marker = 'v', color = "grey", markersize = 10, ls = '', markeredgecolor = 'black', 
        markeredgewidth = 0.2, zorder = 0, label = 'Liu, (1981), concrete')
y_Sklar_bedrockmills = len(data_Sklar_bedrockmills)*[attrition_d]
ax.plot(data_Sklar_bedrockmills['Eimp_Lamb2008.St']/(data_Sklar_bedrockmills['D50']**2), y_Sklar_bedrockmills, marker ='>', color = 'bisque', 
        markersize = 10, markeredgecolor = 'black', markeredgewidth = 0.2, ls = '', label = 'Sklar & Dietrich, (2001), rock')
# y_Sklar_bedrock = [(6*data_Sklar_bedrockdrops['mass'][i]/math.pi*2.65)**(1/3)/1000 for i in range(len(data_Sklar_bedrockdrops))]  # xx check back on grain size
y_Sklar_bedrock = len(data_Sklar_bedrockdrops)*[0.03]
for i in range(len(y_Sklar_bedrock)):
    ax.plot(data_Sklar_bedrockdrops['Eimp'][i]/(y_Sklar_bedrock[i]**2), attrition_d,  marker ='<', 
    ls = '', color = 'dimgrey', markersize = 10, markeredgecolor = 'black', markeredgewidth = 0.2)
ax.plot(data_Sklar_bedrockdrops['Eimp'][0]/(y_Sklar_bedrock[0]**2), attrition_d,  marker ='<', 
        ls = '', color = 'dimgrey', markersize = 10,markeredgecolor = 'black', 
        markeredgewidth = 0.2, zorder = 0, label = 'Sklar & Dietrich, (2004), concrete')
ax.plot(data_energy['Kryžanowski12_specific_energy_J_m2'][0], attrition_d, marker ='p', ls = '', color = 'darkgrey', zorder = 0,
        markersize = 10, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Kryžanowski et al., (2012), concrete') 
y_Scheingross = len(data_Scheingross)*[attrition_d]
ax.plot(data_Scheingross['Eimp_Lamb2008.St']/(data_Scheingross['D50']**2), y_Scheingross, marker ='P', color = '#B0C4DE', 
        markersize = 10, markeredgecolor = 'black', markeredgewidth = 0.2, ls = '', label = 'Scheingross et al., (2014), foam')
ax.plot(data_energy['LitwinMiller21_mean_impact_energy_density_J_m2'][0], attrition_d, ls = '', marker = '^', color = 'khaki',zorder = 0,
     markersize = 10, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Litwin Miller & Jerolmack, (2021), brick and rock')
for i in range(len(data_energy)):
 ax.plot(data_energy['LitwinMiller21_mean_impact_energy_density_J_m2'][i], attrition_d,ls = '',  marker = '^', color = 'khaki',
     markersize = 10, markeredgecolor = 'black', markeredgewidth = 0.2)   
 ax.plot(data_energy['Kryžanowski12_specific_energy_J_m2'][i], attrition_d, marker ='p', ls = '', color = 'darkgrey', markersize = 8, 
         markeredgecolor = 'black', markeredgewidth = 0.2) 

ax.plot(data_Beer['Eeff']/((data_Beer['r.impactor']*2)**2), data_Beer['Diameter_fragment'], marker = 'o', color = "gainsboro", markersize = 10, ls = '', markeredgecolor = 'black', 
        markeredgewidth = 0.2, label ='Beer & Lamb, (2021), artificial sandstone')
    
# Macro-abrasion
for i in range(len(ab)):
    abb = ab['Lithology_ID'].iloc[i]-1
    ax.plot(ke_d_large_xrange2[i], fragment_d/3*2, marker = markers[abb], color = colors[abb], markersize = sizes[abb]+2, ls = '', markeredgecolor = 'black', 
        markeredgewidth = 0.2, zorder = 50)
for i in range(len(data_averaged)):
    if data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]>0:
        ax.plot(ke_d_large_xrange[i], fragment_d, marker = markers[i], color = 'None', markersize = sizes[i]+2, ls = '', markeredgecolor = colors[i], 
        markeredgewidth = 2, zorder = 50)        
for i in range(len(aa)):
    ax.plot(ke_d_small_xrange2[i], fragment_d/3*2, marker = markers[5+i], color = colors[5+i], markersize = sizes[5+i]+2, ls = '', markeredgecolor = 'black', 
        markeredgewidth = 0.2, zorder = 50)
    
# Attrition
for i in range(len(data_averaged)):
    ax.plot(ke_d_large_xrange[i], attrition_d, marker = markers[i], color = colors[i], markersize = sizes[i]+2, ls = '', markeredgecolor = 'black', 
        markeredgewidth = 0.2, zorder = 50) 
for i in range(len(data_averaged)):
    ax.plot(ke_d_small_xrange[i], attrition_d, marker = markers[i], color = colors[i], markersize = sizes[i]+2, ls = '', markeredgecolor = 'black', 
        markeredgewidth = 0.2, zorder = 50)   
ax.plot(ke_d_small_xrange[0], attrition_d, marker = 's', color = 'black', markersize = 10, ls = '', markeredgecolor = 'black', 
        markeredgewidth = 0.2, zorder = 0, label = 'this study')          
# ax.hlines(fragment_d, KE_D_small_min, KE_D_small_max,  color = 'indigo', lw = 5, label = 'small')
# ax.hlines(fragment_d, KE_D_large_min, KE_D_large_max,  color = 'indigo', lw = 5, label = 'large')
# ax.hlines(fragment_d, KE_D_small_min2, KE_D_small_max2,  color = 'violet', lw = 5, label = 'small')
# ax.hlines(fragment_d, KE_D_large_min2, KE_D_large_max2,  color = 'violet', lw = 5, label = 'large')

ax.text(0.1, 0.4, 'micro-abrasion regime', fontsize = 14, transform = ax.transAxes) 
ax.text(0.87, 0.4, 'macro- \nabrasion \nregime', fontsize = 14, transform = ax.transAxes) 

ax.legend(fontsize = 14)
ax.set_ylabel(r'Mean fragment diameter $\mathregular{d_f}$ (m)', fontsize=18, weight = 'bold')
ax.set_xlabel(r'Mean kinetic impact energy density $\mathregular{E_k/D_{i}^2}$ (J/m²)', fontsize=18, weight = 'bold')
ax.set_xscale('log')
ax.set_xlim(0.00007,100000)
ax.set_ylim(-0.001,0.015) 
ax.xaxis.set_major_locator(
    mticker.LogLocator(base=10, subs=(1.0,), numticks=50))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)


fig.tight_layout()
figname = 'Fragment_d_energy_density'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')


#%%#######################
# Size of erosion products

#%% Plot Macro-abrasion - attrition large flume
x_range1 = np.linspace(0,1000,10)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=200)

for i in range(len(data_averaged)):
    ax.plot(data_averaged['Size_corr_attrition_rate_perc_h_1st_run_large_flume'][i]*100, data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
            marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2)

for i in range(len(data_averaged)):
        ax.plot(data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume'][i]*100, data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2)

ax.plot(x_range1, x_range1, lw = 1.5, ls = '--', color = 'black', zorder = 0)
ax.text(0.72, 0.25, '1:1', transform =  ax.transAxes,
        color = 'black', fontsize = 14)
ax.set_xlabel('Attrition rate (%/h)', fontsize=18, weight = 'bold')
ax.set_ylabel('Macro-abrasion rate (%/h)', fontsize=18, weight = 'bold')
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
ax.set_xlim(0,80)
ax.set_ylim(0,200) 

fig.tight_layout()
figname = 'Large_flume_macroabrasion_attrition'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')


#%%############################################################################
# Relate field and flume data

#%% Correct large flume 2nd run attrition rate with average large/small ratio

data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume_corr'] = data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume']/mean_size_corr_attrition_ratio_2nd_large_small
data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume_corr'] = data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume']/mean_size_corr_attrition_ratio_2nd_large_small_g_h
data_averaged['Size_corr_abrasion_rate_perc_h_2nd_run_large_flume_corr'] = data_averaged['Size_corr_abrasion_rate_perc_h_2nd_run_large_flume']/mean_size_corr_abrasion_ratio_2nd_large_small
data_averaged['Size_corr_abrasion_rate_g_h_2nd_run_large_flume_corr'] = data_averaged['Size_corr_abrasion_rate_g_h_2nd_run_large_flume']/mean_size_corr_abrasion_ratio_2nd_large_small_g_h


#%% Plot attrition rates flume and rebound field data - averaged per lithology - large flume data corrected
# get std for abrasion rates
model = LinearRegression()
model.fit(data_averaged[['ISRM_Mean']], data_averaged['Size_corr_attrition_rate_perc_h_small_flume']*100)
predictions = model.predict(data_averaged[['ISRM_Mean']])

fig, ax = plt.subplots(3, 1, figsize=(6, 14), dpi=200)

# Rebound
ax[0].errorbar(data_averaged['ISRM_Mean'][0], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume_corr'][0],
               xerr = data_averaged['ISRM_Std'][i], yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][0],
               marker = markers[0], color = 'None', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 2, ecolor = 'black', elinewidth = 1,
               label = r'large flume')

ax[0].errorbar(data_averaged['ISRM_Mean'][0], data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][0],
                xerr = data_averaged['ISRM_Std'][i], yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][0],
                marker = markers[0], color = 'black', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1,
                label = r'small flume')

for i in range(len(data_averaged)):
    ax[0].errorbar(data_averaged['ISRM_Mean'][i], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume_corr'][i],
                   xerr = data_averaged['ISRM_Std'][i], yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)
    
    ax[0].errorbar(data_averaged['ISRM_Mean'][i], data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][i],
                    xerr = data_averaged['ISRM_Std'][i], yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

# relation with -x2
x_range = np.linspace(1,80,100)
ax[0].plot(x_range, 40*x_range**(-2), color = 'black', lw = 2,
                 ls = '--', label = r'$y = 40x^{-2}$')
ax[0].plot(x_range, model.coef_[0]*x_range+model.intercept_, color = 'black', lw = 2,
                 ls = '-', label = r'y = ' + str(np.round(model.coef_[0],3)) +'x +'+ str(np.round(model.intercept_,3)))

# Fracture intensity
for i in range(len(data_averaged)):
    ax[1].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume_corr'][i],
                    xerr = data_averaged['Intensity_whole_map_main_std'][i], yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)
    
    ax[1].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][i],
                   xerr = data_averaged['Intensity_whole_map_main_std'][i], yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

# Block size
for i in range(len(data_averaged)):
    ax[2].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume_corr'][i],
                    xerr = data_averaged['Blocksize_std_m2_main'][i], yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)
    
    ax[2].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][i],
                    xerr = data_averaged['Blocksize_std_m2_main'][i], yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

ax[1].set_ylabel('Attrition rate (%/h)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis = 'both', which = 'major', labelsize = 14)
ax[0].set_xlim(0,73)
ax[0].set_ylim(0,0.11)
# ax[0].set_yscale('log')
# ax[0].set_xscale('log')
ax[1].set_xlabel(r'Fracture intensity (${\mathregular{m^{-1}}}$)', fontsize=18, weight = 'bold') 
ax[1].tick_params(axis = 'both', which = 'major', labelsize = 14)
ax[1].set_xlim(0,73)
ax[1].set_ylim(0,) 
ax[2].set_xlabel(r'Block size (${\mathregular{m^2}}$)', fontsize=18, weight = 'bold') 
ax[0].set_xlabel('Rebound factor (-)', fontsize=18, weight = 'bold')
ax[2].tick_params(axis = 'both', which = 'major', labelsize = 14)
ax[2].set_xlim(0,)
ax[2].set_ylim(0,) 
ax[0].legend(fontsize = 14)

ax[0].text(0.02, 0.93, 'a)', fontsize = 14, transform = ax[0].transAxes) 
ax[1].text(0.02, 0.93, 'b)', fontsize = 14, transform = ax[1].transAxes) 
ax[2].text(0.02, 0.93, 'c)', fontsize = 14, transform = ax[2].transAxes) 

fig.tight_layout()
figname = 'Attrition_relations_rebound_corr'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot attrition rates flume and rebound field data - averaged per lithology - large flume data corrected, mineral grain sizes

fig, ax = plt.subplots(2, 2, figsize=(14, 14), dpi=200)

# Rebound
ax[0,0].errorbar(data_averaged['ISRM_Mean'][0], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume_corr'][0]*100,
               xerr = data_averaged['ISRM_Std'][i], yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][0]*100,
               marker = markers[0], color = 'None', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 2, ecolor = 'black', elinewidth = 1,
               label = r'large flume')

ax[0,0].errorbar(data_averaged['ISRM_Mean'][0], data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][0]*100,
                xerr = data_averaged['ISRM_Std'][i], yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][0]*100,
                marker = markers[0], color = 'black', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1,
                label = r'small flume')

for i in range(len(data_averaged)):
    ax[0,0].errorbar(data_averaged['ISRM_Mean'][i], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume_corr'][i]*100,
                   xerr = data_averaged['ISRM_Std'][i], yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)
    
    ax[0,0].errorbar(data_averaged['ISRM_Mean'][i], data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][i]*100,
                    xerr = data_averaged['ISRM_Std'][i], yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i]*100,
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

# relation with -x2
x_range = np.linspace(1,80,100)
ax[0,0].plot(x_range, 4000*x_range**(-2), color = 'black', lw = 2,
                 ls = '--', label = r'$y = 4000x^{-2}$')
ax[0,0].plot(x_range, model.coef_[0]*x_range+model.intercept_, color = 'black', lw = 2,
                 ls = '-', label = r'y = ' + str(np.round(model.coef_[0],3)) +'x +'+ str(np.round(model.intercept_,3)))


# Mineral grain size
for i in range(len(data_averaged)):
    ax[0,1].errorbar(data_averaged['Mineral_D50_mum'][i], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume_corr'][i]*100,
                    xerr = data_averaged['Mineral_weighted_std_mum'][i], yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)
    
    ax[0,1].errorbar(data_averaged['Mineral_D50_mum'][i], data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][i]*100,
                    xerr = data_averaged['Mineral_weighted_std_mum'][i], yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

# Fracture intensity
for i in range(len(data_averaged)):
    ax[1,0].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume_corr'][i]*100,
                    xerr = data_averaged['Intensity_whole_map_main_std'][i], yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)
    
    ax[1,0].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][i]*100,
                   xerr = data_averaged['Intensity_whole_map_main_std'][i], yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

# Block size
for i in range(len(data_averaged)):
    ax[1,1].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_attrition_rate_perc_h_2nd_run_large_flume_corr'][i]*100,
                    xerr = data_averaged['Blocksize_std_m2_main'][i], yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)
    
    ax[1,1].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_attrition_rate_perc_h_small_flume'][i]*100,
                    xerr = data_averaged['Blocksize_std_m2_main'][i], yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

fig.supylabel('Attrition rate (%/h)', fontsize= 20, weight = 'bold')
ax[0,0].set_xlabel('Rebound factor (-)', fontsize= 20, weight = 'bold')
ax[0,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0,0].set_xlim(0,73)
ax[0,0].set_ylim(0,11)
ax[0,0].legend(fontsize = 16)
ax[0,1].set_xlabel(r'Mineral $\mathregular{D_{50}}$ (${\mathregular{\mu}}$m)', fontsize= 20, weight = 'bold') 
ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0,1].set_xlim(0,500)
ax[0,1].set_ylim(0,11) 
ax[1,0].set_xlabel(r'Fracture intensity (${\mathregular{m^{-1}}}$)', fontsize= 20, weight = 'bold') 
ax[1,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[1,0].set_xlim(0,73)
ax[1,0].set_ylim(0,11) 
ax[1,1].set_xlabel(r'Block size (${\mathregular{m^2}}$)', fontsize= 20, weight = 'bold') 
ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[1,1].set_xlim(0,)
ax[1,1].set_ylim(0,11) 

ax[0,0].text(0.02, 0.95, 'a)', fontsize = 16, transform = ax[0,0].transAxes) 
ax[0,1].text(0.02, 0.95, 'b)', fontsize = 16, transform = ax[0,1].transAxes) 
ax[1,0].text(0.02, 0.95, 'c)', fontsize = 16, transform = ax[1,0].transAxes) 
ax[1,1].text(0.02, 0.95, 'd)', fontsize = 16, transform = ax[1,1].transAxes) 

fig.tight_layout()
figname = 'Attrition_relations_rebound_corr_2_2'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot attrition rates flume and comp, tensile strength field data - averaged per lithology
# divide by the number of pebbles*(pi D^2)
# Initial_no_1st_run_large_flume
fig, ax = plt.subplots(2,1, figsize=(12, 14), dpi=200)

# Compressive strength
for i in range(len(data_averaged)): 
    ax[0].errorbar(data_averaged['Compressive_strength_Pa'][i], data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'][i]/(data_averaged['Initial_no_large_flume'][i]*(math.pi*0.07**2)),
                   xerr = data_averaged['Compressive_strength_Pa_std'][i], #yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1, zorder = 50)
    
    ax[0].errorbar(data_averaged['Compressive_strength_Pa'][i], data_averaged['Size_corr_attrition_rate_g_h_small_flume'][i]/(data_averaged['Initial_no_small_flume'][i]*(math.pi*0.025**2)),
                    xerr = data_averaged['Compressive_strength_Pa_std'][i], #yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1, zorder = 50)
    
ax[0].plot(data_literature_erosion_tensile_add['Kryžanowski12_compressive_strength_90d_MPa'] , data_literature_erosion_tensile_add['Kryžanowski12_erosion_rate_g_h'],
           marker ='P', ls = '', color = 'darkgrey', markersize = 8,  markeredgecolor = 'black', markeredgewidth = 0.2,label = 'Kryžanowski et al., (2012), concrete')   
ax[0].errorbar(data_Turowski['Compressive strength [MPa]'] , data_Turowski['ERCS mean [g/h]'],
           xerr = data_Turowski['Compressive strength SEM [MPa]'], yerr = data_Turowski['ERCS SD [g/h]'], 
           marker ='o', ls = '', color = 'burlywood', markersize = 7, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Turowski et al., (2023), rock')

ax[0].errorbar(data_averaged['Compressive_strength_Pa'][0], data_averaged['Size_corr_attrition_rate_g_h_small_flume'][0]/(data_averaged['Initial_no_small_flume'][0]*(math.pi*0.025**2)),
                xerr = data_averaged['Compressive_strength_Pa_std'][i], #yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                marker = markers[i], color = 'black', markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1, zorder = 0, label = 'this study')

# any relation?
# relation with -x2
# x_range = np.linspace(0.05,300,100)
# ax[0].plot(x_range, 600*x_range**(-2),
#                   ls = '-', color = 'black', lw = 2)
ax[0].grid()
# ax[0].text(0.1, 0.8, r'y = 600x$^{-2}$', fontsize = 14, transform = ax[0].transAxes) 

# Tensile strength
for i in range(len(data_averaged)):
    ax[1].errorbar(data_averaged['Tensile_strength_Pa'][i]/10**6, data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'][i]/(data_averaged['Initial_no_large_flume'][i]*(math.pi*0.07**2)),
                    xerr = data_averaged['Tensile_strength_Pa_std'][i]/10**6, #yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1, zorder = 50)
    
    ax[1].errorbar(data_averaged['Tensile_strength_Pa'][i]/10**6, data_averaged['Size_corr_attrition_rate_g_h_small_flume'][i]/(data_averaged['Initial_no_small_flume'][i]*(math.pi*0.025**2)),
                    xerr = data_averaged['Tensile_strength_Pa_std'][i]/10**6,#yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1, zorder = 50)

ax[1].plot(data_literature_erosion_tensile['Sklar01_rock_Log10_Scale_tensile'] , data_literature_erosion_tensile['Sklar01_rock_Log10_Scale_erosion_rate'],
           marker ='>', ls = '', color = 'bisque', markersize = 8,  markeredgecolor = 'black', markeredgewidth = 0.2,label = 'Sklar & Dietrich, (2001), rock')
ax[1].plot(data_literature_erosion_tensile['Sklar01_concrete_Log10_Scale_tensile'] , data_literature_erosion_tensile['Sklar01_concrete_Log10_Scale_erosion_rate'],
           marker ='<', ls = '', color = 'dimgrey', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sklar & Dietrich, (2004), concrete')
ax[1].plot(data_literature_erosion_tensile['Helbig12_concrete_Log10_Scale_tensile'] , data_literature_erosion_tensile['Helbig12_concrete_Log10_Scale_erosion_rate'],
           marker ='X', ls = '', color = 'darkgrey', markersize = 8,  markeredgecolor = 'black', markeredgewidth = 0.2,label = 'Helbig et al., (2012), concrete')
ax[1].plot(data_literature_erosion_tensile['Scheingross14_foam_Log10_Scale_tensile'] , data_literature_erosion_tensile['Scheingross14_foam_Log10_Scale_erosion_rate'],
           marker ='P', ls = '', color = '#B0C4DE', markersize = 8,  markeredgecolor = 'black', markeredgewidth = 0.2,label = 'Scheingross et al., (2014), foam')
ax[1].plot(data_literature_erosion_tensile['Auel14_concrete_Log10_Scale_tensile'] , data_literature_erosion_tensile['Auel14_concrete_Log10_Scale_erosion_rate'],
           marker ='*', ls = '', color = 'lightgrey', markersize = 9, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Auel, (2014), concrete')
ax[1].errorbar(data_Turowski['Tensile strength [N/mmÂ²]'], data_Turowski['ERCS mean [g/h]'], 
               xerr = data_Turowski['Tensile strength SEM [N/mmÂ²]'], yerr = data_Turowski['ERCS SD [g/h]'], 
               ls = "", marker = 'o', color = 'burlywood', markersize = 7, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Turowski et al., (2023), rock') # 1N/mm2 = 1 MPa
# ax[1].plot(data_energy['LitwinMiller21_tensile_strength_Pa']/1000000, data_energy['LitwinMiller21_attrition_rate_s2_m2'], # needs conversion - time is not available, this is mass loss per energy
#            marker ='8', ls = '', color = 'peru', markersize = 9, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Litwin Miller & Jerolmack, (2021), brick and rock')
ax[1].errorbar(data_averaged['Tensile_strength_Pa'][0]/10**6, data_averaged['Size_corr_attrition_rate_g_h_small_flume'][0]/(data_averaged['Initial_no_small_flume'][0]*(math.pi*0.025**2)),
                xerr = data_averaged['Tensile_strength_Pa_std'][i]/10**6,#yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                marker = markers[i], color = 'black', markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1, zorder = 0, label = 'this study')

# add Opalinus clay
# relation with -x2
# x_range = np.linspace(0.05,80,100)
# ax[1].plot(x_range, 100*x_range**(-2),
#                   ls = '-', color = 'black', lw = 2)
ax[1].grid()
# ax[1].text(0.2, 0.68, r'y = 100x$^{-2}$', fontsize = 14, transform = ax[1].transAxes) 

ax[0].legend(loc='lower left', fontsize = 20, facecolor = 'white', framealpha = 1)
ax[1].legend(loc='lower left', fontsize = 20, facecolor = 'white', framealpha = 1)
fig.supylabel('Attrition rate (g/h)', fontsize = 25, weight = 'bold')
# ax[1].set_ylabel('Attrition rate (g/h)', fontsize = 25, weight = 'bold')
ax[0].set_xlabel(r'Compressive strength (MPa)', fontsize = 25, weight = 'bold') 
ax[0].tick_params(axis = 'both', which = 'major', labelsize = 20)
ax[0].set_xlim(7,300)
ax[0].set_ylim(0.0001,100000) 
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[1].set_xlabel('Tensile strength (MPa)', fontsize = 25, weight = 'bold')
ax[1].tick_params(axis = 'both', which = 'major', labelsize = 20)
ax[1].set_xlim(0.05,40)
ax[1].set_ylim(0.0001,100000)
ax[1].set_yscale('log')
ax[1].set_xscale('log')

ax[0].text(0.02, 0.95, 'a)', fontsize = 20, transform = ax[0].transAxes) 
ax[1].text(0.02, 0.95, 'b)', fontsize = 20, transform = ax[1].transAxes) 

fig.tight_layout()
figname = 'Attrition_relations_g_h_literature'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot attrition rate vs mean kinetic impact energy density
fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi=200)

for i in range(len(data_averaged)): 
    ax.plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i]**2, data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'][i],
                    marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)
    
    ax.plot(KE_small_mean/data_averaged['Tensile_strength_Pa'][i]**2, data_averaged['Size_corr_attrition_rate_g_h_small_flume'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)
    
ax.legend(loc='lower right', fontsize = 14, facecolor = 'white', framealpha = 1)
ax.set_ylabel('Attrition rate (g/h)', fontsize=18, weight = 'bold')
ax.set_xlabel(r'Normalized mean kinetic impact energy $\mathregular{E_k/\sigma_t^2}$ (J/Pa²)', fontsize=18, weight = 'bold')
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax.set_xlim(7,300)
# ax.set_ylim(0.0001,10) 
ax.set_yscale('log')
ax.set_xscale('log')

fig.tight_layout()
figname = 'Energy_attrition_rate'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')


#%% Plot attrition rates flume and comp, tensile strength field data, erodibility - averaged per lithology 
fig, ax = plt.subplots(2,2, figsize=(17, 12), dpi=200)

# Compressive strength
ax[0,0].errorbar(data_averaged['Compressive_strength_Pa'][0], data_averaged['Size_corr_attrition_rate_g_h_small_flume'][0],
                xerr = data_averaged['Compressive_strength_Pa_std'][i], #yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                marker = markers[i], color = 'black', markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1, zorder = 50, label = 'this study')

for i in range(len(data_averaged)): 
    ax[0,0].errorbar(data_averaged['Compressive_strength_Pa'][i], data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'][i],
                   xerr = data_averaged['Compressive_strength_Pa_std'][i], #yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1, zorder = 50)
    
    ax[0,0].errorbar(data_averaged['Compressive_strength_Pa'][i], data_averaged['Size_corr_attrition_rate_g_h_small_flume'][i],
                    xerr = data_averaged['Compressive_strength_Pa_std'][i], #yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1, zorder = 50)
    
ax[0,0].plot(data_literature_erosion_tensile_add['Kryžanowski12_compressive_strength_90d_MPa'] , data_literature_erosion_tensile_add['Kryžanowski12_erosion_rate_g_h'],
           marker ='p', ls = '', color = 'darkgrey', markersize = 8,  markeredgecolor = 'black', markeredgewidth = 0.2,label = 'Kryžanowski et al., (2012), concrete')   
ax[0,0].errorbar(data_Turowski['Compressive strength [MPa]'] , data_Turowski['ERCS mean [g/h]'],
           xerr = data_Turowski['Compressive strength SEM [MPa]'], yerr = data_Turowski['ERCS SD [g/h]'], 
           marker ='o', ls = '', color = 'burlywood', markersize = 7, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Turowski et al., (2023), rock')

# any relation?
# relation with -x2
# x_range = np.linspace(0.05,300,100)
# ax[0,0].plot(x_range, 600*x_range**(-2),
#                   ls = '-', color = 'black', lw = 2)
ax[0,0].grid()
# ax[0,0].text(0.1, 0.8, r'y = 600x$^{-2}$', fontsize = 14, transform = ax[0,0].transAxes) 

# Tensile strength
ax[0,1].errorbar(data_averaged['Tensile_strength_Pa'][0]/10**6, data_averaged['Size_corr_attrition_rate_g_h_small_flume'][0],
                xerr = data_averaged['Tensile_strength_Pa_std'][i]/10**6,#yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                marker = markers[i], color = 'black', markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1, zorder = 0, label = 'this study')

for i in range(len(data_averaged)):
    ax[0,1].errorbar(data_averaged['Tensile_strength_Pa'][i]/10**6, data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'][i],
                    xerr = data_averaged['Tensile_strength_Pa_std'][i]/10**6, #yerr = large_flume_sec_run_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1, zorder = 50)
    
    ax[0,1].errorbar(data_averaged['Tensile_strength_Pa'][i]/10**6, data_averaged['Size_corr_attrition_rate_g_h_small_flume'][i],
                    xerr = data_averaged['Tensile_strength_Pa_std'][i]/10**6,#yerr = small_flume_data_average_std['Size_corr_attrition_rate_perc_h_std'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1, zorder = 50)

ax[0,1].plot(data_literature_erosion_tensile['Sklar01_rock_Log10_Scale_tensile'] , data_literature_erosion_tensile['Sklar01_rock_Log10_Scale_erosion_rate'],
           marker ='>', ls = '', color = 'bisque', markersize = 8,  markeredgecolor = 'black', markeredgewidth = 0.2,label = 'Sklar & Dietrich, (2001), rock')
ax[0,1].plot(data_literature_erosion_tensile['Sklar01_concrete_Log10_Scale_tensile'] , data_literature_erosion_tensile['Sklar01_concrete_Log10_Scale_erosion_rate'],
           marker ='<', ls = '', color = 'dimgrey', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sklar & Dietrich, (2004), concrete')
ax[0,1].plot(data_literature_erosion_tensile['Helbig12_concrete_Log10_Scale_tensile'] , data_literature_erosion_tensile['Helbig12_concrete_Log10_Scale_erosion_rate'],
           marker ='X', ls = '', color = 'darkgrey', markersize = 8,  markeredgecolor = 'black', markeredgewidth = 0.2,label = 'Helbig et al., (2012), concrete')
ax[0,1].plot(data_literature_erosion_tensile['Scheingross14_foam_Log10_Scale_tensile'] , data_literature_erosion_tensile['Scheingross14_foam_Log10_Scale_erosion_rate'],
           marker ='P', ls = '', color = '#B0C4DE', markersize = 8,  markeredgecolor = 'black', markeredgewidth = 0.2,label = 'Scheingross et al., (2014), foam')
ax[0,1].plot(data_literature_erosion_tensile['Auel14_concrete_Log10_Scale_tensile'] , data_literature_erosion_tensile['Auel14_concrete_Log10_Scale_erosion_rate'],
           marker ='*', ls = '', color = 'lightgrey', markersize = 9, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Auel, (2014), concrete')
ax[0,1].errorbar(data_Turowski['Tensile strength [N/mmÂ²]'], data_Turowski['ERCS mean [g/h]'], 
               xerr = data_Turowski['Tensile strength SEM [N/mmÂ²]'], yerr = data_Turowski['ERCS SD [g/h]'], 
               ls = "", marker = 'o', color = 'burlywood', markersize = 7, markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Turowski et al., (2023), rock') # 1N/mm2 = 1 MPa

# add Opalinus clay
# relation with -x2
# x_range = np.linspace(0.05,80,100)
# ax[0,1].plot(x_range, 100*x_range**(-2),
#                   ls = '-', color = 'black', lw = 2)
ax[0,1].grid()
# ax[0,1].text(0.2, 0.68, r'y = 100x$^{-2}$', fontsize = 14, transform = ax[0,1].transAxes) 

# Energy
for i in range(len(data_averaged)): 
    ax[1,0].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i]**2, data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'][i],
                    marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)    
    ax[1,0].plot(KE_small_mean/data_averaged['Tensile_strength_Pa'][i]**2, data_averaged['Size_corr_attrition_rate_g_h_small_flume'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)
   
ax[1,0].plot(KE_small_mean/data_averaged['Tensile_strength_Pa'][0]**2, data_averaged['Size_corr_attrition_rate_g_h_small_flume'][0],
                marker = markers[0], color = 'black', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 0, label = r'small flume' )
ax[1,0].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][0]**2, data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'][0],
                marker = markers[0], color = 'None', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 2, zorder = 0, label = r'large flume' )

# Energy
for i in range(len(data_averaged)): 
    ax[1,1].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i], data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'][i],
                    marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)    
    ax[1,1].plot(KE_small_mean/data_averaged['Tensile_strength_Pa'][i], data_averaged['Size_corr_attrition_rate_g_h_small_flume'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)
    ax[1,1].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i], data_averaged['Size_corr_macro_abrasion_rate_g_h_1st_run_large_flume'][i],
                      marker = markers[i], color = 'black', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)    
    ax[1,1].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i], data_averaged['Size_corr_macro_abrasion_rate_g_h_2nd_run_large_flume'][i],
                      marker = markers[i], color = 'black', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)    
    ax[1,1].plot(KE_small_mean/data_averaged['Tensile_strength_Pa'][i], data_averaged['Size_corr_macro_abrasion_rate_g_h_small_flume'][i],
                      marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 2, zorder = 50)

ax[1,1].plot(KE_small_mean/data_averaged['Tensile_strength_Pa'][0], data_averaged['Size_corr_attrition_rate_g_h_small_flume'][0],
                marker = markers[0], color = 'black', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 0, label = r'small flume' )
ax[1,1].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][0], data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume'][0],
                marker = markers[0], color = 'None', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 2, zorder = 0, label = r'large flume' )


ax[0,0].legend(loc='lower left', fontsize = 14, facecolor = 'white', framealpha = 1)
ax[0,1].legend(loc='lower left', fontsize = 14, facecolor = 'white', framealpha = 1)
fig.supylabel('Attrition rate (g/h)', fontsize=18, weight = 'bold')
# ax[0,1].set_ylabel('Attrition rate (g/h)', fontsize=18, weight = 'bold')
ax[0,0].set_xlabel(r'Compressive strength (MPa)', fontsize=18, weight = 'bold') 
ax[0,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0,0].set_xlim(7,300)
ax[0,0].set_ylim(0.0001,100000) 
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,1].set_xlabel('Tensile strength (MPa)', fontsize=18, weight = 'bold')
ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0,1].set_xlim(0.05,40)
ax[0,1].set_ylim(0.0001,100000)
ax[0,1].set_yscale('log')
ax[0,1].set_xscale('log')
ax[1,0].legend(loc='lower right', fontsize = 14, facecolor = 'white', framealpha = 1)
ax[1,0].set_xlabel(r'Normalized mean kinetic impact energy $\mathregular{E_k/\sigma_t^2}$ (J/Pa²)', fontsize=18, weight = 'bold')
ax[1,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax[1,0].set_xlim(7,300)
# ax[1,0].set_ylim(0.0001,10) 
ax[1,0].set_yscale('log')
ax[1,0].set_xscale('log')
ax[1,1].legend(loc='lower right', fontsize = 14, facecolor = 'white', framealpha = 1)
ax[1,1].set_xlabel(r'Normalized mean kinetic impact energy $\mathregular{E_k/\sigma_t}$ (J/Pa)', fontsize=18, weight = 'bold')
# ax[1,1].set_xlabel(r'Erodibility (MPa)', fontsize=18, weight = 'bold') 
ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax[1,1].set_xlim(7,300)
# ax[1,1].set_xlim(1,4000) 
ax[1,1].set_yscale('log')
ax[1,1].set_xscale('log')

ax[0,0].text(0.02, 0.95, 'a)', fontsize = 14, transform = ax[0,0].transAxes) 
ax[0,1].text(0.02, 0.95, 'b)', fontsize = 14, transform = ax[0,1].transAxes) 
ax[1,0].text(0.02, 0.95, 'c)', fontsize = 14, transform = ax[1,0].transAxes) 
ax[1,1].text(0.02, 0.95, 'd)', fontsize = 14, transform = ax[1,1].transAxes) 

fig.tight_layout()
figname = 'Attrition_relations_erodibility_g_h_literature'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')

#%% Plot attrition rates flume and comp, energy - averaged per lithology 
data_averaged['Norm_energy_small_J_Pa2'] = KE_small_mean/data_averaged['Tensile_strength_Pa']**2
data_averaged['Norm_energy_large_J_Pa2'] = KE_large_mean/data_averaged['Tensile_strength_Pa']**2
data_averaged['Norm_energy_small_J_Pa'] = KE_small_mean/data_averaged['Tensile_strength_Pa']
data_averaged['Norm_energy_large_J_Pa'] = KE_large_mean/data_averaged['Tensile_strength_Pa']

data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume_vol_corr']=data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume']/(data_averaged['Initial_no_large_flume'][i]*(math.pi*0.07**2))
data_averaged['Size_corr_attrition_rate_g_h_small_flume_vol_corr']=data_averaged['Size_corr_attrition_rate_g_h_small_flume']/(data_averaged['Initial_no_small_flume'][i]*(math.pi*0.025**2))
data_averaged['Size_corr_macro_abrasion_rate_g_h_2nd_run_large_flume_vol_corr']=data_averaged['Size_corr_macro_abrasion_rate_g_h_2nd_run_large_flume']/(data_averaged['Initial_no_large_flume'][i]*(math.pi*0.07**2))
data_averaged['Size_corr_macro_abrasion_rate_g_h_small_flume_vol_corr']=data_averaged['Size_corr_macro_abrasion_rate_g_h_small_flume']/(data_averaged['Initial_no_small_flume'][i]*(math.pi*0.025**2))

aa_df = pd.melt(data_averaged, value_vars=['Size_corr_attrition_rate_g_h_2nd_run_large_flume_vol_corr', 'Size_corr_attrition_rate_g_h_small_flume_vol_corr',
               'Size_corr_macro_abrasion_rate_g_h_2nd_run_large_flume_vol_corr', 'Size_corr_macro_abrasion_rate_g_h_small_flume_vol_corr'],                  
                 value_name='Rates')
aa_df['Norm_energy_J_Pa2'] = pd.concat([data_averaged['Norm_energy_large_J_Pa2'], data_averaged['Norm_energy_small_J_Pa2'] ,
                                        data_averaged['Norm_energy_large_J_Pa2'], data_averaged['Norm_energy_small_J_Pa2'] ], ignore_index=True)
aa_df['Norm_energy_J_Pa'] = pd.concat([data_averaged['Norm_energy_large_J_Pa'], data_averaged['Norm_energy_small_J_Pa'] , 
                                       data_averaged['Norm_energy_large_J_Pa'], data_averaged['Norm_energy_small_J_Pa'] ], ignore_index=True)

# Linear regression
# Squared 
mask = aa_df['Norm_energy_J_Pa2'].notna() & aa_df['Rates'].notna()
X_raw = aa_df.loc[mask, 'Norm_energy_J_Pa2'].values
y_raw = aa_df.loc[mask, 'Rates'].values
X_2d = X_raw.reshape(-1, 1)
model_lin = LinearRegression(fit_intercept=False)
model_lin.fit(X_2d, y_raw)
x_smooth = np.logspace(np.log10(X_raw[X_raw > 0].min()), 
                        np.log10(X_raw.max()), 100).reshape(-1, 1)
y_smooth = model_lin.predict(x_smooth)
r_squared = model_lin.score(X_2d, y_raw)

# Single 
mask_single = aa_df['Norm_energy_J_Pa'].notna() & aa_df['Rates'].notna()
X_raw_single = aa_df.loc[mask_single, 'Norm_energy_J_Pa'].values
y_raw_single = aa_df.loc[mask_single, 'Rates'].values
X_2d_single = X_raw_single.reshape(-1, 1)
model_lin_single = LinearRegression(fit_intercept=False)
model_lin_single.fit(X_2d_single, y_raw_single)
x_smooth_single = np.logspace(np.log10(X_raw_single[X_raw_single > 0].min()), 
                       np.log10(X_raw_single.max()), 100).reshape(-1, 1)
y_smooth_single = model_lin_single.predict(x_smooth_single)
r_squared_single = model_lin_single.score(X_2d_single, y_raw_single)

# Power law
# Squared
mask = (aa_df['Norm_energy_J_Pa2'] > 0) & (aa_df['Rates'] > 0)
x_raw = aa_df.loc[mask, 'Norm_energy_J_Pa2'].values
y_raw = aa_df.loc[mask, 'Rates'].values
log_x = np.log10(x_raw).reshape(-1, 1)
log_y = np.log10(y_raw)
model = LinearRegression(fit_intercept=True)
model.fit(log_x, log_y)
b = model.coef_[0]
a = 10**model.intercept_   
r2 = model.score(log_x, log_y)
x_fit = np.logspace(np.log10(x_raw.min()), np.log10(x_raw.max()), 100).reshape(-1, 1)
log_y_fit = model.predict(np.log10(x_fit))
y_fit = 10**log_y_fit

# Single
mask_single = (aa_df['Norm_energy_J_Pa'] > 0) & (aa_df['Rates'] > 0)
x_single_raw = aa_df.loc[mask_single, 'Norm_energy_J_Pa'].values
y_single_raw = aa_df.loc[mask_single, 'Rates'].values
log_x_single = np.log10(x_single_raw).reshape(-1, 1)
log_y_single = np.log10(y_single_raw)
model_single = LinearRegression(fit_intercept=True)
model_single.fit(log_x_single, log_y_single)
b_single = model_single.coef_[0]
a_single = 10**model_single.intercept_
r2_single = model_single.score(log_x_single, log_y_single)
x_fit_single = np.logspace(np.log10(x_single_raw.min()), 
                               np.log10(x_single_raw.max()), 100).reshape(-1, 1)
log_y_predicted = model_single.predict(np.log10(x_fit_single))
y_fit_single = 10**log_y_predicted

#%% Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
# Energy
for i in range(len(data_averaged)): 
    ax[0].plot(data_averaged['Norm_energy_large_J_Pa2'][i], data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume_vol_corr'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)    
    ax[0].plot(data_averaged['Norm_energy_small_J_Pa2'][i], data_averaged['Size_corr_attrition_rate_g_h_small_flume_vol_corr'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)
    
    ax[0].plot(data_averaged['Norm_energy_large_J_Pa2'][i], data_averaged['Size_corr_macro_abrasion_rate_g_h_2nd_run_large_flume_vol_corr'][i],
                      marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)    
    ax[0].plot(data_averaged['Norm_energy_small_J_Pa2'][i], data_averaged['Size_corr_macro_abrasion_rate_g_h_small_flume_vol_corr'][i],
                      marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)

ax[0].plot(x_fit, y_fit, color='black',ls ="-")
ax[0].text(2e-16, 1050, f"$y = {a:.2e} \cdot x^{{{b:.2f}}}$", fontsize = 14)
ax[0].text(2e-16, 600, f"$R^2={r2:.2f}$", fontsize = 14)

# Energy
for i in range(len(data_averaged)): 
    ax[1].plot(data_averaged['Norm_energy_large_J_Pa'][i],  data_averaged['Size_corr_attrition_rate_g_h_2nd_run_large_flume_vol_corr'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)    
    ax[1].plot(data_averaged['Norm_energy_small_J_Pa'][i],  data_averaged['Size_corr_attrition_rate_g_h_small_flume_vol_corr'][i],
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)
       
    ax[1].plot(data_averaged['Norm_energy_large_J_Pa'][i], data_averaged['Size_corr_macro_abrasion_rate_g_h_2nd_run_large_flume_vol_corr'][i],
                      marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)    
    ax[1].plot(data_averaged['Norm_energy_small_J_Pa'] [i], data_averaged['Size_corr_macro_abrasion_rate_g_h_small_flume_vol_corr'][i],
                      marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)

# ax[1].plot(KE_small_mean/data_averaged['Tensile_strength_Pa'][0], data_averaged['Size_corr_attrition_rate_g_h_small_flume'][0]/(data_averaged['Initial_no_large_flume'][0]*(math.pi*0.07**2)),
#                 marker = markers[0], color = 'black', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 0, label = r'Attrition' )
# ax[1].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][0], data_averaged['Size_corr_macro_abrasion_rate_g_h_small_flume'][0]/(data_averaged['Initial_no_small_flume'][0]*(math.pi*0.02**2)),
#                 marker = markers[0], color = 'None', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 2, zorder = 0, label = r'Macro-abrasion' )

# ax[1].plot(x_smooth_single, y_smooth_single, color='black',ls ="-")
# ax[1].text(2e-9, 100, f"y = {model_lin_single.coef_[0]:.2e}x", fontsize = 14)
# ax[1].text(2e-9, 50, 'R2 = ' + str(np.round(r_squared_single,2)), fontsize = 14)

ax[1].plot(x_fit_single, y_fit_single, color='black',ls ="-")
ax[1].text(2e-9, 1050, f"$y = {a_single:.2e} \cdot x^{{{b_single:.2f}}}$", fontsize = 14)
ax[1].text(2e-9, 600, f"$R^2={r2_single:.2f}$", fontsize = 14)

# ax[0].legend(loc='lower right', fontsize = 14, facecolor = 'white', framealpha = 1)
ax[1].legend(loc='lower right', fontsize = 14, facecolor = 'white', framealpha = 1)
ax[0].set_ylabel('Abrasion rate (g/h)', fontsize=18, weight = 'bold')
fig.supxlabel(r'Normalized mean kinetic impact energy', fontsize=18, weight = 'bold')
ax[0].set_xlabel(r'$\mathregular{E_k/\sigma_t^2}$ (J/Pa²)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax[0].set_xlim(7,300)
# ax[0].set_ylim(0.0001,10) 
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[1].set_xlabel(r'$\mathregular{E_k/\sigma_t}$ (J/Pa)', fontsize=18, weight = 'bold')
# ax[1].set_xlabel(r'Erodibility (MPa)', fontsize=18, weight = 'bold') 
ax[1].tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax[1].set_xlim(7,300)
# ax[1].set_xlim(1,4000) 
ax[1].set_yscale('log')
ax[1].set_xscale('log')

ax[0].text(0.02, 0.95, 'b)', fontsize = 14, transform = ax[0].transAxes) 
ax[1].text(0.02, 0.95, 'c)', fontsize = 14, transform = ax[1].transAxes) 

fig.tight_layout()
figname = 'Abrasion_relations_energy_g_h_literature'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot macro-abrasion rate vs mean kinetic impact energy density
fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi=200)

for i in range(len(data_averaged)): 
    ax.plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i]**2, data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i],
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)
    ax.plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i]**2, data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i],
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)
   
    # ax.plot(KE_small_mean/data_averaged['Tensile_strength_Pa'][i]**2, data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i],
    #                 marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)
    
    
ax.legend(loc='lower right', fontsize = 14, facecolor = 'white', framealpha = 1)
ax.set_ylabel('Macro-abrasion rate (%/h)', fontsize=18, weight = 'bold')
ax.set_xlabel(r'Normalized mean kinetic impact energy $\mathregular{E_k/\sigma_t^2}$ (J/Pa²)', fontsize=18, weight = 'bold')
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax.set_xlim(7,300)
# ax.set_ylim(0.0001,10) 
ax.set_yscale('log')
ax.set_xscale('log')

fig.tight_layout()
figname = 'Energy_macro_abrasion_rate'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')


#%% Plot macro_abrasion rates flume and comp, tensile strength field data - averaged per lithology 
# get std for abrasion rates

fig, ax = plt.subplots(3, 2, figsize=(14, 20), dpi=200)

# Compressive strength
for i in range(len(data_averaged)):
    ax[0,0].errorbar(data_averaged['Compressive_strength_Pa'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['Compressive_strength_Pa_std'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1) 
    ax[0,0].errorbar(data_averaged['Compressive_strength_Pa'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                   xerr = data_averaged['Compressive_strength_Pa_std'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "None", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
    # ax[0,0].errorbar(data_averaged['Comp_strength_mean'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i],
    #                 xerr = data_averaged['Comp_strength_std'][i], yerr = small_flume_data_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i],
    #                 marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

# Tensile strength
for i in range(len(data_averaged)):
    ax[0,1].errorbar(data_averaged['Tensile_strength_Pa'][i]/10**6, data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['Tensile_strength_Pa_std'][i]/10**6,
                  marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)  
    ax[0,1].errorbar(data_averaged['Tensile_strength_Pa'][i]/10**6, data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                    xerr = data_averaged['Tensile_strength_Pa_std'][i]/10**6, yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "None", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
    # ax[0,1].errorbar(data_averaged['Tensile_strength_Pa'][i]/10**6, data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i],
    #                 xerr = data_averaged['Tensile_strength_Pa_std'][i]/10**6, yerr = small_flume_data_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i],
    #                 marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)


# Fracture intensity
for i in range(len(data_averaged)):
    ax[1,0].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['Intensity_whole_map_main_std'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)   
    ax[1,0].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                    xerr = data_averaged['Intensity_whole_map_main_std'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "None", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
                         # ax[1,0].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i],
    #                xerr = data_averaged['Intensity_whole_map_main_std'][i], yerr = small_flume_data_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i],
    #                marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

# Block size
for i in range(len(data_averaged)):
    ax[1,1].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['Blocksize_std_m2_main'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)   
    ax[1,1].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                    xerr = data_averaged['Blocksize_std_m2_main'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                  marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "None", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
    # ax[1,1].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i],
    #                 xerr = data_averaged['Blocksize_std_m2_main'][i], yerr = small_flume_data_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i],
    #                marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

# Energy
for i in range(len(data_averaged)): 
    ax[2,0].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i]**2, data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)    
    ax[2,0].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i]**2, data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)
    
# Erodibility
# for i in range(len(data_averaged)): 
#     ax[2,1].plot(data_averaged['Erodibility_Turowski24_large'][i]/1e6, data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
#                    marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)    
#     ax[2,1].plot(data_averaged['Erodibility_Turowski24_large'][i]/1e6, data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
#                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)
 
# ax[2,1].plot(data_averaged['Erodibility_Turowski24_large'][0]/1e6, data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][0]*100,
#                marker = markers[0], color = 'None', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 2, zorder = 0, label = r'$1^{\rm{st}}$ run' )
# ax[2,1].plot(data_averaged['Erodibility_Turowski24_large'][0]/1e6, data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][0]*100,
#                marker = markers[0], color = 'black', markersize = sizes[0], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 0, label = r'$2^{\rm{nd}}$ run')
    
fig.supylabel('Macro-abrasion rate (%/h)', fontsize=18, weight = 'bold')
ax[0,0].set_xlabel(r'Compressive strength (MPa)', fontsize=18, weight = 'bold') 
ax[0,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0,0].set_xlim(0,)
ax[0,0].set_ylim(1,400) 
ax[0,0].set_yscale('log')
ax[0,1].set_xlabel('Tensile strength (MPa)', fontsize=18, weight = 'bold')
ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0,1].set_xlim(0,10)
ax[0,1].set_ylim(1,400) 
ax[0,1].set_yscale('log')
ax[1,0].set_xlabel(r'Fracture intensity (${\mathregular{m^{-1}}}$)', fontsize=18, weight = 'bold') 
ax[1,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[1,0].set_xlim(0,73)
ax[1,0].set_ylim(1,400) 
ax[1,0].set_yscale('log')
ax[1,1].set_xlabel(r'Block size (${\mathregular{m^2}}$)', fontsize=18, weight = 'bold') 
ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[1,1].set_xlim(0,0.27)
ax[1,1].set_ylim(1,400) 
ax[1,1].set_yscale('log')
ax[2,0].set_xlabel(r'Normalized mean kinetic impact energy $\mathregular{E_k/\sigma_t^2}$ (J/Pa²)', fontsize=18, weight = 'bold')
ax[2,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax[2,0].set_xlim(0)
ax[2,0].set_ylim(1,400)  
ax[2,0].set_xscale('log')
ax[2,0].set_yscale('log')
# ax[2,1].set_xlabel(r'Erodibility (MPa)', fontsize=18, weight = 'bold')
ax[2,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax[2,0].set_xlim(0)
ax[2,1].set_ylim(1,400)  
ax[2,1].set_xscale('log')
ax[2,1].set_yscale('log')
ax[2,1].legend(loc= 'lower right', fontsize = 16)

ax[0,0].text(0.02, 0.95, 'a)', fontsize = 16, transform = ax[0,0].transAxes) 
ax[0,1].text(0.02, 0.95, 'b)', fontsize = 16, transform = ax[0,1].transAxes) 
ax[1,0].text(0.02, 0.95, 'c)', fontsize = 16, transform = ax[1,0].transAxes) 
ax[1,1].text(0.02, 0.95, 'd)', fontsize = 16, transform = ax[1,1].transAxes) 
ax[2,0].text(0.02, 0.95, 'e)', fontsize = 16, transform = ax[2,0].transAxes) 
ax[2,1].text(0.02, 0.95, 'f)', fontsize = 16, transform = ax[2,1].transAxes)  

fig.tight_layout()
figname = 'Macro_abrasion_relations_perc_h'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')

#%% Plot macro_abrasion rates flume and rebound field data - averaged per lithology 
# get std for abrasion rates

fig, ax = plt.subplots(3, 3, figsize=(20, 20), dpi=200)

# Rebound factor
for i in range(len(data_averaged)):
    ax[0,0].errorbar(data_averaged['ISRM_Mean'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['ISRM_Std'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1) 
    ax[0,0].errorbar(data_averaged['ISRM_Mean'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                   xerr = data_averaged['ISRM_Std'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
   
# Fracture intensity
for i in range(len(data_averaged)):
    ax[0,1].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['Intensity_whole_map_main_std'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)   
    ax[0,1].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                    xerr = data_averaged['Intensity_whole_map_main_std'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
    
# Block size
for i in range(len(data_averaged)):
    ax[0,2].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['Blocksize_std_m2_main'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)   
    ax[0,2].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                    xerr = data_averaged['Blocksize_std_m2_main'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                  marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
   
# Amygdules
for i in range(len(data_averaged)):
    ax[1,0].errorbar( data_thin_sections['Diameter_amygdules_mean_cm'][i],data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_thin_sections['Distance_amygdules_cm_std'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1) 
    ax[1,0].errorbar(data_thin_sections['Diameter_amygdules_mean_cm'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                xerr = data_thin_sections['Distance_amygdules_cm_std'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)

for i in range(len(data_averaged)):
    ax[1,1].errorbar( data_thin_sections['Perc_area_amygdules'][i],data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_thin_sections['Std_area_amygdules_um2'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1) 
    ax[1,1].errorbar(data_thin_sections['Perc_area_amygdules'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                xerr = data_thin_sections['Std_area_amygdules_um2'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)

# Mineral grain size
for i in range(len(data_averaged)):
    ax[1,2].errorbar( data_averaged['Mineral_D50_mum'][i],data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['Mineral_weighted_std_mum'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1) 
    ax[1,2].errorbar(data_averaged['Mineral_D50_mum'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                xerr = data_averaged['Mineral_weighted_std_mum'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
   
# Energy
for i in range(len(data_averaged)): 
    ax[2,0].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)    
    ax[2,0].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)

fig.supylabel('Macro-abrasion rate (%/h)', fontsize=18, weight = 'bold')
ax[0,0].set_xlabel(r'Rebound factor (-)', fontsize=18, weight = 'bold') 
ax[0,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0,0].set_xlim(0,)
ax[0,0].set_ylim(1,400) 
ax[0,0].set_yscale('log')
ax[0,1].set_xlabel(r'Fracture intensity (segments/${\mathregular{m^2}}$)', fontsize=18, weight = 'bold') 
ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0,1].set_xlim(0,73)
ax[0,1].set_ylim(1,400) 
ax[0,1].set_yscale('log')
ax[0,2].set_xlabel(r'Block size (${\mathregular{m^2}}$)', fontsize=18, weight = 'bold') 
ax[0,2].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0,2].set_xlim(0,0.27)
ax[0,2].set_ylim(1,400) 
ax[0,2].set_yscale('log')
ax[1,0].set_xlabel(r'Mean diameter amygdules (cm)', fontsize=18, weight = 'bold') 
ax[1,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[1,0].set_xlim(0,0.14) 
ax[1,0].set_ylim(1,400)  
ax[1,0].set_yscale('log')
ax[1,1].set_xlabel(r'Relative surface area amygdules (%)', fontsize=18, weight = 'bold')
ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[1,1].set_xlim(0,18)
ax[1,1].set_ylim(1,400) 
ax[1,1].set_yscale('log')
ax[1,2].set_xlabel(r'Mineral $\mathregular{D_{50}}$ (${\mathregular{\mu}}$m)', fontsize=18, weight = 'bold')
ax[1,2].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[1,2].set_xlim(0,500)
ax[1,2].set_ylim(1,400) 
ax[1,2].set_yscale('log')
ax[2,0].set_xlabel(r'Norm. mean kinetic impact energy $\mathregular{E_k/\sigma_t}$ (J/Pa)', fontsize=18, weight = 'bold')
ax[2,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax[2,0].set_xlim(0)
ax[2,0].set_ylim(1,400)  
# ax[2,0].set_xscale('log')
ax[2,0].set_yscale('log')
# ax[2,0].set_xlim(0)

ax[0,0].text(0.02, 0.95, 'a)', fontsize = 16, transform = ax[0,0].transAxes) 
ax[0,1].text(0.02, 0.95, 'b)', fontsize = 16, transform = ax[0,1].transAxes) 
ax[0,2].text(0.02, 0.95, 'c)', fontsize = 16, transform = ax[0,2].transAxes) 
ax[1,0].text(0.02, 0.95, 'd)', fontsize = 16, transform = ax[1,0].transAxes) 
ax[1,1].text(0.02, 0.95, 'e)', fontsize = 16, transform = ax[1,1].transAxes) 
ax[2,0].text(0.02, 0.95, 'f)', fontsize = 16, transform = ax[1,2].transAxes)  
ax[2,1].text(0.02, 0.95, 'g)', fontsize = 16, transform = ax[2,0].transAxes) 

fig.tight_layout()
figname = 'Macro_abrasion_relations_rebound_amygdules_grain_size_perc_h'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot macro_abrasion rates flume and rebound field data - averaged per lithology 
# get std for abrasion rates

fig, ax = plt.subplots(3, 2, figsize=(14, 20), dpi=200)

# Rebound factor
for i in range(len(data_averaged)):
    ax[0,0].errorbar(data_averaged['ISRM_Mean'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['ISRM_Std'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1) 
    ax[0,0].errorbar(data_averaged['ISRM_Mean'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                   xerr = data_averaged['ISRM_Std'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
    # ax[0,0].errorbar(data_averaged['Comp_strength_mean'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i],
    #                 xerr = data_averaged['Comp_strength_std'][i], yerr = small_flume_data_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i],
    #                 marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

# Energy
for i in range(len(data_averaged)): 
    ax[0,1].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, zorder = 50)    
    ax[0,1].plot(KE_large_mean/data_averaged['Tensile_strength_Pa'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                    marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 50)
   
# Fracture intensity
for i in range(len(data_averaged)):
    ax[1,0].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['Intensity_whole_map_main_std'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)   
    ax[1,0].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                    xerr = data_averaged['Intensity_whole_map_main_std'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                   marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
                         # ax[1,0].errorbar(data_averaged['Intensity_whole_map_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i],
    #                xerr = data_averaged['Intensity_whole_map_main_std'][i], yerr = small_flume_data_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i],
    #                marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

# Block size
for i in range(len(data_averaged)):
    ax[1,1].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_averaged['Blocksize_std_m2_main'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1)   
    ax[1,1].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                    xerr = data_averaged['Blocksize_std_m2_main'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                  marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)
    # ax[1,1].errorbar(data_averaged['Blocksize_mean_m2_main'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_small_flume'][i],
    #                 xerr = data_averaged['Blocksize_std_m2_main'][i], yerr = small_flume_data_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i],
    #                marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = 'black', markeredgewidth = 0.2, elinewidth = 1)

 
# Amygdules
for i in range(len(data_averaged)):
    ax[2,0].errorbar( data_thin_sections['Diameter_amygdules_mean_cm'][i],data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_thin_sections['Distance_amygdules_cm_std'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1) 
    ax[2,0].errorbar(data_thin_sections['Diameter_amygdules_mean_cm'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                xerr = data_thin_sections['Distance_amygdules_cm_std'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)

for i in range(len(data_averaged)):
    ax[2,1].errorbar( data_thin_sections['Perc_area_amygdules'][i],data_averaged['Size_corr_macro_abrasion_rate_perc_h_1st_run_large_flume'][i]*100,
                    xerr = data_thin_sections['Std_area_amygdules_um2'][i], 
                   marker = markers[i], color = 'None', markersize = sizes[i], ls = '', markeredgecolor = colors[i], markeredgewidth = 2, ecolor = colors[i], elinewidth = 1) 
    ax[2,1].errorbar(data_thin_sections['Perc_area_amygdules'][i], data_averaged['Size_corr_macro_abrasion_rate_perc_h_2nd_run_large_flume'][i]*100,
                xerr = data_thin_sections['Std_area_amygdules_um2'][i], yerr = large_flume_sec_run_average_std['Size_corr_macro_abrasion_rate_perc_h_std'][i]*100,
                marker = markers[i], color = colors[i], markersize = sizes[i], ls = '', markeredgecolor = "black", markeredgewidth = 0.2, ecolor = colors[i], elinewidth = 1)


fig.supylabel('Macro-abrasion rate (%/h)', fontsize=18, weight = 'bold')
ax[0,0].set_xlabel(r'Rebound factor (-)', fontsize=18, weight = 'bold') 
ax[0,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0,0].set_xlim(0,)
# ax[0,0].set_ylim(1,400) 
ax[0,0].set_yscale('log')
ax[0,1].set_xlabel(r'Norm. mean kinetic impact energy $\mathregular{E_k/\sigma_t}$ (J/Pa)', fontsize=18, weight = 'bold')
ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax[0,1].set_xlim(0)
ax[0,1].set_ylim(1,400)  
# ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
# ax[0,1].set_xlim(0)
ax[1,0].set_xlabel(r'Fracture intensity (${\mathregular{m^{-1}}}$)', fontsize=18, weight = 'bold') 
ax[1,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[1,0].set_xlim(0,73)
ax[1,0].set_ylim(1,400) 
ax[1,0].set_yscale('log')
ax[1,1].set_xlabel(r'Block size (${\mathregular{m^2}}$)', fontsize=18, weight = 'bold') 
ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[1,1].set_xlim(0,0.27)
ax[1,1].set_ylim(1,400) 
ax[1,1].set_yscale('log')
ax[2,0].set_xlabel(r'Mean diameter amygdules (cm)', fontsize=18, weight = 'bold') 
ax[2,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[2,0].set_xlim(0,0.14) 
ax[2,0].set_ylim(1,400)  
ax[2,0].set_yscale('log')
ax[2,1].set_xlabel(r'Relative surface area amygdules (%)', fontsize=18, weight = 'bold')
ax[2,1].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[2,1].set_xlim(0,18)
ax[2,1].set_ylim(1,400) 
ax[2,1].set_yscale('log')

ax[0,0].text(0.02, 0.95, 'a)', fontsize = 16, transform = ax[0,0].transAxes) 
ax[0,1].text(0.02, 0.95, 'b)', fontsize = 16, transform = ax[0,1].transAxes) 
ax[1,0].text(0.02, 0.95, 'c)', fontsize = 16, transform = ax[1,0].transAxes) 
ax[1,1].text(0.02, 0.95, 'd)', fontsize = 16, transform = ax[1,1].transAxes) 
ax[2,0].text(0.02, 0.95, 'e)', fontsize = 16, transform = ax[2,0].transAxes) 
ax[2,1].text(0.02, 0.95, 'f)', fontsize = 16, transform = ax[2,1].transAxes)  

fig.tight_layout()
figname = 'Macro_abrasion_relations_rebound_amygdules_perc_h'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Upscaling
# scaling Y/(kv sigma2t)
data_averaged['Ee_small'] = data_averaged['Size_corr_attrition_rate_perc_h_small_flume']/16*6.6e-10 
data_averaged['Ee_large']  = data_averaged['Size_corr_attrition_rate_perc_h_small_flume']/80*6.6e-10

rho_s = 2650 
rho_w = 1000
R_b =(rho_s-rho_w)/rho_w
g = 9.81

# River properties
W = 20 # channel width
R_h = 50 # Hydraulic radius
S = 0.001 # Slope
Q_s = 45 # Sediment transport
D_s = 0.3 # Sediment grain size

tau_star = (R_h*S)/(R_b*D_s)
tau_star_c = 23 # critical shear stress
u_star = (g*R_h*S)**(-0.5)
w_f = 4 # grain fall velocity in still water
Q_c = 5.7*rho_s*W*(R_b*g*D_s**3)**(-0.5)*(tau_star-tau_star_c)**(3/2)
E_small = 0.08*R_b*g*data_averaged['Ee_small']*Q_s*(tau_star/tau_star_c-1)**(-0.5)*(1-Q_s/Q_c)*(1-(u_star/w_f)**2)**(3/2)

#%% Plot Schmidt hammer literature data
# data_literature_Schmidt_L = data_literature_Schmidt[data_literature_Schmidt['Schmidt_hammer_procedure'].str.contains('L', na=False)]
data_literature_Schmidt_L = data_literature_Schmidt[~data_literature_Schmidt['Schmidt_hammer_procedure'].str.contains('N', na=False)]
# data_literature_Schmidt_L = data_literature_Schmidt_L[data_literature_Schmidt_L['Schmidt_hammer_procedure'].str.contains('IRSM', na=False)]

color_dict = {'Plutonic': 'teal', 'Volcanic':'cornflowerblue', 'Limestone': 'darkorange', 'Sedimentary': 'tan',
              'Metamorphic': 'brown', 'Mineral':'darkorange'}
markerss = {'Plutonic': 'o',  'Volcanic':'s', 'Limestone': 'v', 'Sedimentary': '^', 'Metamorphic': 'D', 'Mineral':'p'}
rock_types = data_literature_Schmidt_L['Lithology_type'].unique()

fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

for rock in rock_types:
    subset = data_literature_Schmidt_L[data_literature_Schmidt_L['Lithology_type'] == rock]
    
    ax[0].errorbar(
        subset['UCS_mean_MPa'], 
        subset['Rebound_mean'],
        xerr=subset['UCS_std_MPa'], 
        yerr=subset['Rebound_std'],
        marker=markerss[rock],  
        color=color_dict[rock], 
        ls='', 
        markersize=5, 
        alpha=0.8, 
        label=rock,           
        markeredgecolor='black', 
        markeredgewidth=0.2, 
        elinewidth=1
    )
      
    ax[1].errorbar(
         subset['Youngs_modulus_mean_GPa'], 
         subset['Rebound_mean'],
         xerr=subset['Youngs_modulus_std_GPa'], 
         yerr=subset['Rebound_std'],
         marker=markerss[rock],  
         color=color_dict[rock], 
         ls='', 
         markersize=5, 
         alpha=0.8, 
         label=rock,           
         markeredgecolor='black', 
         markeredgewidth=0.2, 
         elinewidth=1
     )
    
ax[0].set_ylabel('Rebound value (-)', fontsize=18, weight = 'bold')
ax[0].set_xlabel('Uniaxial compressive strength (MPa)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[0].set_xlim(0,)
ax[0].set_ylim(0,)
ax[1].set_xlabel(r'Young modulus (GPa)', fontsize=18, weight = 'bold') 
ax[1].tick_params(axis = 'both', which = 'major', labelsize = 16)
ax[1].set_xlim(0,)
ax[1].set_ylim(0,) 
ax[0].legend(fontsize = 16)

ax[0].text(0.02, 0.95, 'a)', fontsize = 16, transform = ax[0].transAxes) 
ax[1].text(0.02, 0.95, 'b)', fontsize = 16, transform = ax[1].transAxes) 

fig.tight_layout()
figname = 'Schmidt_hammer_literature'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')


























