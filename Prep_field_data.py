# -*- coding: utf-8 -*-
"""
This script prepares and analyses field data (Schmidt hammer, fracture spacing and 
optical thin section analysis) and is used as input for "Figures_article.py"

Related to article submitted to Earth Surface Processes and Landforms

Step II after the analysis of thin sections 

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

#%% Load packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

#%% Load data
# Use only field measurements close to sampling locations - "outcrop"
# replace filepath with your directory
all_data = pd.read_csv(r'filepath\All_field_data_article_outcrops.csv')
outpath_figures = r'filepath\Figures'
data_ts = pd.read_csv(r'filepath\Data_thin_sections.csv') 
grain_size = pd.read_csv(r'filepath\Mineral_grain_sizes.csv')  
porosity = pd.read_csv(r'filepath\Porosity.csv') 

#%% Prepare data close to outcrops 

data = all_data[~all_data['Maplabel'].isin(['Cbs', 'CTpb', 'CRb','Mkf', 'TeB', 'Dtr'])] # drop lithologies in other catchments
data = data[~((data['Maplabel'] == 'Mnu') & (data['Date'] == '23/08/2024'))] # drop Mnu in Gooseberry 
data = data[~ ((data['Date'] == '31/07/2024'))] # drop Mng in upper bedrock section
# data = data[~((data['Maplabel'] == 'Mni') & (data['Longitude'] > -92.008))] # drop all Icelandite except those at the sampling location, others are in Lester or downstream
# data = data[~((data['Maplabel'] == 'Mni') & (data['Latitude'] > 46.847))] # drop all Icelandite except those at the sampling location, others are in Lester or downstream
data = data[~((data['Maplabel'] == 'Mnu') & (data['Latitude'] > 46.85))] # drop all Mnu except those at the sampling location, others are in Lester or downstream
data = data[~((data['Maplabel'] == 'Mnu') & (data['Latitude'] < 46.839))] # drop all Mnu except those at the sampling location, others are in Lester or downstream
data = data[~((data['Maplabel'] == 'Mnu') & (data['Longitude'] < -92.006))] # drop all Mnu except those at the sampling location, others are in Lester or downstream
data = data.reset_index(drop = True)

#%% Prepare data
# Unit lithologies (including top and bottom of flows, specific flows) - relate to the samples measured in the flume
lithologies = ['Mns', 'Mis', 'Mnu_Bottom', 'Mng_Bottom', 'Bottom_flow_4', 'Mnu_Top', 'Mng_Top', 'Top_flow_3', 'Mni', 'Cst']
lithologies_label = ['Diabase Mls', 'Diabase Mli', 'Mnu Bottom', 'Mng Bottom', 'Mng #4 Bottom', 'Mnu Top', 'Mng Top', 'Mng #3 Top', 'Icelandite', 'Sandstone']
lithologies_ID = np.arange(1,11,1)

# add lithologies to data
lithologyy = []
for i in range(len(data)):
    if data['Bedrock_unit'].iloc[i] == 'Top_flow_3':
        aa = 'Top_flow_3'
        lithologyy.append(aa)    
    elif data['Bedrock_unit'].iloc[i] == 'Bottom_flow_4':
        aa = 'Bottom_flow_4'
        lithologyy.append(aa)
    elif data['Unit_lithology'].iloc[i] is not np.nan:
        aa = data['Unit_lithology'].iloc[i]
        lithologyy.append(aa)
    elif data['Unit_lithology'].iloc[i] is np.nan:
        aa = data['Maplabel'].iloc[i]
        lithologyy.append(aa)
data['Lithology'] = lithologyy

colors = ['peru', 'sienna', 'yellowgreen', 'dodgerblue', 'cornflowerblue', 'forestgreen', 'royalblue', 'blue', 'gold', 'violet']
markers = ['s', 's', 'v', 'v', 'v', '^', '^', '^', 'D', 'p']
sizes = [10,10,11,11,11,11,11,11,10,11]

#%% Prepare grain size data
# Get median (D50) and std for each thin section
grain_size_classes_upper= [31.2, 62.5, 125, 250, 500, 1000, 2000] # mum, upper limits
size_data = grain_size.iloc[:, 1:8]
weights = size_data.values

# Cumsum and get D50 using upper limits
cum_pct = size_data.cumsum(axis=1).div(size_data.sum(axis=1), axis=0) * 100
grain_size['Mineral_D50'] = [np.interp(50, cum_pct.iloc[i], grain_size_classes_upper) for i in range(len(grain_size))]

# Get standard deviation (weighted for the classes), use midpoints
# Calculate Midpoints
lower_limits = [15] + grain_size_classes_upper[:-1] 
midpoints = [(l + u) / 2 for l, u in zip(lower_limits, grain_size_classes_upper)]
means = [np.average(midpoints, weights=w) for w in weights]
grain_size['Mineral_weighted_std'] = [
    np.sqrt(np.average((midpoints - means[i])**2, weights=weights[i])) 
    for i in range(len(grain_size))]

#%% Plot grain size data
custom_palette = sns.color_palette(colors)
porosity['Lithology_ID'] = pd.Categorical(
    porosity['Lithology_ID'], 
    categories=lithologies_ID, 
    ordered=True
)

fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=200)

sns.pointplot(data=porosity, x='Lithology_ID', y='Porosity_perc', 
              join=False, markersize = 9, capsize=0.1, errorbar='sd', palette=custom_palette, hue = 'Lithology_ID',
              markeredgecolor = 'black', markeredgewidth = 0.2)

sns.stripplot(data=porosity, x='Lithology_ID', y='Porosity_perc', 
              color='gray', size = 7, jitter=True, ax=ax, zorder=1)

ax.set_xlabel('Lithology', fontsize=18, weight = 'bold')
ax.set_ylabel(r'Porosity (%)', fontsize=18, weight = 'bold')
ax.set_xticks(range(len(lithologies_label)))
ax.set_xticklabels(lithologies_label, rotation=30, ha='right', fontsize=16)
ax.tick_params(axis = 'x', which = 'major', labelsize = 16, rotation = 30)
ax.tick_params(axis = 'y', which = 'major', labelsize = 16)
ax.set_ylim(0,) 

fig.tight_layout()
figname = 'Porosity'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot grain size data
custom_palette = sns.color_palette(colors)
grain_size['Lithology_ID'] = pd.Categorical(
    grain_size['Lithology_ID'], 
    categories=lithologies_ID, 
    ordered=True
)

fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=200)

sns.pointplot(data=grain_size, x='Lithology_ID', y='Mineral_D50', 
              join=False, markersize = 9, capsize=0.1, errorbar='sd', palette=custom_palette, hue = 'Lithology_ID',
              markeredgecolor = 'black', markeredgewidth = 0.2)
# sns.stripplot(data=grain_size, x='Lithology_ID', y='Mineral_D50', 
#               color='black', jitter=True, palette=custom_palette, hue = 'Lithology_ID')

sns.stripplot(data=grain_size, x='Lithology_ID', y='Mineral_D50', 
              color='gray', size = 7, jitter=True, ax=ax, zorder=1)

ax.set_xlabel('Lithology', fontsize=18, weight = 'bold')
ax.set_ylabel(r'$\mathregular{D_{50}}$ (µm)', fontsize=18, weight = 'bold')
ax.set_xticks(range(len(lithologies_label)))
ax.set_xticklabels(lithologies_label, rotation=30, ha='right', fontsize=16)
ax.tick_params(axis = 'x', which = 'major', labelsize = 16, rotation = 30)
ax.tick_params(axis = 'y', which = 'major', labelsize = 16)
ax.set_ylim(0,) 

fig.tight_layout()
figname = 'Mineral_grain_size'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot histogram of compressive strength 
rebound_mean_lithology = []
rebound_std_lithology = []
rebound_values_lithology = []
rebound_meas_per_lithology = []
smaller_10_lithology = []
for i in range(len(lithologies)):
    aa = data.iloc[:,39:64][data['Lithology']==lithologies[i]]
    ab = aa.stack()
    ab = ab.dropna()
    ab = pd.DataFrame(ab.reset_index(drop=True))
    ab['Lithology'] = [lithologies[i]]*len(ab)
    am = aa.mean().mean()
    astd = aa.stack().std()
    rebound_values_lithology.append(ab)
    rebound_mean_lithology.append(am)
    rebound_std_lithology.append(astd)
    rebound_meas_per_lithology.append(len(ab))
rebound_values_lithology = pd.concat(rebound_values_lithology)
rebound_values_lithology.columns = ['Rebound', 'Lithology']   
 
data_lithology = pd.DataFrame([pd.Series(lithologies), pd.Series(rebound_mean_lithology), 
                      pd.Series(rebound_std_lithology)]).transpose()

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=200)
custom_palette = sns.color_palette(colors)
sns.boxplot(data=rebound_values_lithology, x='Lithology', y = 'Rebound', palette=custom_palette, hue = 'Lithology')

for i in range(len(rebound_meas_per_lithology)):
    ax.text(i, 113, 'n=' + str(rebound_meas_per_lithology[i]), fontsize = 14, ha='center')
    # ax.text(i, 105, 'n<10=' + str(smaller_10_lithology[i]), fontsize = 14, ha='center')
    
ax.set_xlabel('Lithology', fontsize=18, weight = 'bold')
ax.set_ylabel('Rebound value (-)', fontsize=18, weight = 'bold')
ax.tick_params(axis = 'both', which = 'major', labelsize = 14, rotation = 30)
ax.set_ylim(0,120) 

fig.tight_layout()
figname = 'Lithologies_rebound'
# fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot histogram of compressive strength 
comp_strength_mean_lithology = []
comp_strength_std_lithology = []
comp_strength_values_lithology = []
comp_strength_meas_per_lithology = []
smaller_10_lithology = []
for i in range(len(lithologies)):
    aa = data.iloc[:,65:90][data['Lithology']==lithologies[i]]
    ab = aa.stack()
    ab = ab.dropna()
    ab = pd.DataFrame(ab.reset_index(drop=True))
    ab['Lithology'] = [lithologies[i]]*len(ab)
    am = aa.mean().mean()
    astd = aa.stack().std()
    comp_strength_values_lithology.append(ab)
    comp_strength_mean_lithology.append(am)
    comp_strength_std_lithology.append(astd)
    comp_strength_meas_per_lithology.append(len(ab))
    smaller_10_lithology.append(int(sum(data.iloc[:,64][data['Lithology']==lithologies[i]])))
comp_strength_values_lithology = pd.concat(comp_strength_values_lithology)
comp_strength_values_lithology.columns = ['Comp_strength', 'Lithology']   
 
data_lithology = pd.DataFrame([pd.Series(lithologies), pd.Series(comp_strength_mean_lithology), 
                      pd.Series(comp_strength_std_lithology)]).transpose()

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=200)
custom_palette = sns.color_palette(colors)
sns.boxplot(data=comp_strength_values_lithology, x='Lithology', y = 'Comp_strength', palette=custom_palette, hue = 'Lithology')

for i in range(len(comp_strength_meas_per_lithology)):
    ax.text(i, 113, 'n=' + str(comp_strength_meas_per_lithology[i]), fontsize = 14, ha='center')
    ax.text(i, 106, r'$\mathregular{n_{<10}}$=' + str(smaller_10_lithology[i]), fontsize = 14, ha='center')
    # ax.text(i, 105, 'n<10=' + str(smaller_10_lithology[i]), fontsize = 14, ha='center')
    
ax.set_xlabel('Lithology', fontsize=18, weight = 'bold')
ax.set_ylabel('Compressive strength (MPa)', fontsize=18, weight = 'bold')
ax.tick_params(axis = 'both', which = 'major', labelsize = 14, rotation = 30)
ax.set_ylim(0,120) 

fig.tight_layout()
figname = 'Lithologies_comp_strength'
# fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Plot histogram of mean block size 
block_size_mean_lithology = []
block_size_std_lithology = []
block_size_values_lithology = []
blocksize_meas_per_lithology = []
for i in range(len(lithologies)):
    aa = data['Blocksize_mean_m2_main'][data['Lithology']==lithologies[i]]
    ab = aa
    ab = ab.dropna()
    ab = pd.DataFrame(ab.reset_index(drop=True))
    ab['Lithology'] = [lithologies[i]]*len(ab)
    am = aa.mean().mean()
    astd = aa.std()
    block_size_values_lithology.append(ab)
    block_size_mean_lithology.append(am)
    block_size_std_lithology.append(astd)
    blocksize_meas_per_lithology.append(len(ab))
block_size_values_lithology = pd.concat(block_size_values_lithology)
block_size_values_lithology.columns = ['Blocksize_mean_m2_main', 'Lithology']   
block_size_values_lithology = block_size_values_lithology.reset_index(drop = True)
data_blocksize_lithology = pd.DataFrame([pd.Series(lithologies), pd.Series(block_size_mean_lithology), 
                      pd.Series(block_size_std_lithology)]).transpose()

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
custom_palette = sns.color_palette(colors)
sns.boxplot(data=block_size_values_lithology, x='Lithology', y = 'Blocksize_mean_m2_main', palette=custom_palette, hue = 'Lithology')

for i in range(len(blocksize_meas_per_lithology)):
    ax.text(i, 0.5, 'n=' + str(blocksize_meas_per_lithology[i]), fontsize = 14, ha='center')
    
ax.set_xlabel('Lithology', fontsize=18, weight = 'bold')
ax.set_ylabel('Mean block size (m²)', fontsize=18, weight = 'bold')
ax.tick_params(axis = 'both', which = 'major', labelsize = 14, rotation = 30)
ax.set_ylim(0,0.55) 

fig.tight_layout()
figname = 'Lithologies_mean_block_size'
# fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')


#%% Plot histogram of fracture intensity 
frac_intensity_mean_lithology = []
frac_intensity_std_lithology = []
frac_intensity_values_lithology = []
frac_intensity_meas_per_lithology = []
for i in range(len(lithologies)):
    aa = data['Intensity_whole_map_main'][data['Lithology']==lithologies[i]]
    aa = aa.dropna()
    ab = aa
    ab = ab.dropna()
    ab = pd.DataFrame(ab.reset_index(drop=True))
    ab['Lithology'] = [lithologies[i]]*len(ab)
    am = aa.mean().mean()
    astd = aa.std()
    frac_intensity_values_lithology.append(ab)
    frac_intensity_mean_lithology.append(am)
    frac_intensity_std_lithology.append(astd)
    frac_intensity_meas_per_lithology.append(len(ab))
frac_intensity_values_lithology = pd.concat(frac_intensity_values_lithology)
frac_intensity_values_lithology.columns = ['Intensity_whole_map_main', 'Lithology']   
frac_intensity_values_lithology =frac_intensity_values_lithology.reset_index(drop = True)
data_intensity_lithology = pd.DataFrame([pd.Series(lithologies), pd.Series(frac_intensity_mean_lithology), 
                      pd.Series(frac_intensity_std_lithology)]).transpose()

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
custom_palette = sns.color_palette(colors)
sns.boxplot(data=frac_intensity_values_lithology, x='Lithology', y = 'Intensity_whole_map_main', palette=custom_palette, hue='Lithology',)

for i in range(len(frac_intensity_meas_per_lithology)):
    ax.text(i, 52, 'n=' + str(frac_intensity_meas_per_lithology[i]), fontsize = 14, ha='center')
    
ax.set_xlabel('Lithology', fontsize=18, weight = 'bold')
ax.set_ylabel('Fracture intensity (${\mathregular{m^{-1}}}$)', fontsize=18, weight = 'bold')
ax.tick_params(axis = 'both', which = 'major', labelsize = 14, rotation = 30)
ax.set_ylim(0,55) 

fig.tight_layout()
figname = 'Lithologies_fracture_intensity'
# fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Figure article

median_distance_mean = data_ts['Median_distance_amygdules'].mean()
median_distance_std = data_ts['Median_distance_amygdules'].std()

labelsizes = 16
fontsizes = 18

fig, ax = plt.subplots(3,2, figsize=(20, 18), dpi=200)
# a) Compressive strength
custom_palette = sns.color_palette(colors)
axs = ax[0,0]
sns.boxplot(data=comp_strength_values_lithology, x='Lithology', y = 'Comp_strength', palette=custom_palette, hue = 'Lithology', ax = axs)

for i in range(len(comp_strength_meas_per_lithology)):
    ax[0,0].text(i/10+0.05, 0.95, 'n=' + str(comp_strength_meas_per_lithology[i]), fontsize = labelsizes, ha='center', transform=ax[0,0].transAxes)
    # ax[0,0].text(i/10, 0.9, r'$\mathregular{n_{<10}}$=' + str(smaller_10_lithology[i]), fontsize = labelsizes-2, ha='center', transform=ax[0,0].transAxes)
    # ax[0,0].text(i, 105, 'n<10=' + str(smaller_10_lithology[i]), fontsize = labelsizes, ha='center')
    
ax[0,0].set_ylabel('Compressive strength (MPa)', fontsize=fontsizes, weight = 'bold')
ax[0,0].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[0,0].set_xticklabels([])
ax[0,0].set_ylim(0,125) 

# b) Compressive strength - 
custom_palette = sns.color_palette(colors)
axs = ax[0,1]
ax[0,1].bar(lithologies, smaller_10_lithology, color = colors, edgecolor = 'black')

ax[0,1].set_ylabel(r'$\mathregular{N < 10}$', fontsize=fontsizes, weight = 'bold')
ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[0,1].set_xticklabels([])
ax[0,1].set_ylim(0,) 

# c) Frac intensity
axs = ax[1,0]
sns.boxplot(data=frac_intensity_values_lithology, x='Lithology', y = 'Intensity_whole_map_main', palette=custom_palette, hue='Lithology', ax = axs)

for i in range(len(frac_intensity_meas_per_lithology)):
    ax[1,0].text(i/10+0.05, 0.95, 'n=' + str(frac_intensity_meas_per_lithology[i]), fontsize = labelsizes, ha='center', transform=ax[1,0].transAxes)
    
ax[1,0].set_ylabel('Fracture intensity (${\mathregular{m^{-1}}}$)', fontsize=fontsizes, weight = 'bold')
ax[1,0].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[1,0].set_xticklabels([])
ax[1,0].set_ylim(0,55) 

# d) Block size 
axs = ax[1,1]
sns.boxplot(data=block_size_values_lithology, x='Lithology', y = 'Blocksize_mean_m2_main', palette=custom_palette, hue = 'Lithology', ax = axs)

for i in range(len(blocksize_meas_per_lithology)):
    ax[1,1].text(i/10+0.05, 0.95, 'n=' + str(blocksize_meas_per_lithology[i]), fontsize = labelsizes, ha='center', transform=ax[1,1].transAxes)
    
ax[1,1].set_ylabel('Mean block size (m²)', fontsize=fontsizes, weight = 'bold')
ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[1,1].set_ylim(0,0.55) 
ax[1,1].set_xticklabels([])

# e) thin sections
colorss = [ 'royalblue', 'forestgreen']
lithology = ['Mng_Top', 'Mnb_Top']
color_map = dict(zip(lithology, colorss))

ax[2,0].axhline(median_distance_mean, 0, 9.5, ls = '--', color = 'black', lw = 2, label = 'Mean')   
ax[2,0].axhspan(median_distance_mean-median_distance_std, median_distance_mean+median_distance_std, 0, 9.5, ls = 'None', color = 'lightgrey', lw = 2, label = 'Mean')   
for i, row in data_ts.iterrows():
    current_lith = row['Lithology']
    current_color = color_map.get(current_lith, 'grey') 
    
    ax[2,0].errorbar(row['Perc_area_amygdules'], 
                row['Median_distance_amygdules'],
                yerr = row['Distance_amygdules_std'],
                marker = '^', 
                markersize = 11, 
                color = current_color, # Applied here
                ls = 'None',
                mec = 'black', # Optional: adds a black edge for better visibility
                mew = 0.5)
legend_elements = [
    Line2D([0], [0], marker='^', color='w', label='Mng Top',
           markerfacecolor='royalblue', markersize=12, markeredgecolor='black'),
    Line2D([0], [0], marker='^', color='w', label='Mnb Top',
           markerfacecolor='forestgreen', markersize=12, markeredgecolor='black')
]

ax[2,0].legend(handles=legend_elements, loc='upper right', fontsize=labelsizes, frameon=True)    
ax[2,0].set_xlabel(r'Relative surface area amygdules (%)', fontsize=fontsizes, weight = 'bold')        
ax[2,0].set_ylabel(r'Median distance amygdules ($\mathregular{\mu}$m)', fontsize=fontsizes, weight = 'bold')
ax[2,0].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[2,0].set_xlim(0,) 
ax[2,0].set_ylim(0,) 

# Figure
ax[0,0].text(-0.1, 0.95, 'a)', fontsize = labelsizes, transform = ax[0,0].transAxes) 
ax[0,1].text(-0.07, 0.95, 'b)', fontsize = labelsizes, transform = ax[0,1].transAxes)
ax[1,0].text(-0.1, 0.95, 'c)', fontsize = labelsizes, transform = ax[1,0].transAxes)
ax[1,1].text(-0.07, 0.95, 'd)', fontsize = labelsizes, transform = ax[1,1].transAxes) 
ax[2,0].text(-0.1, 0.95, 'e)', fontsize = labelsizes, transform = ax[2,0].transAxes)
# fig.supxlabel('Lithology', fontsize=18, weight = 'bold')

fig.tight_layout()
figname = 'Lithologies_properties_amygdules'
# fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Figure article

median_distance_mean = data_ts['Median_distance_amygdules'].mean()
median_distance_std = data_ts['Median_distance_amygdules'].std()

labelsizes = 16
fontsizes = 18

fig, ax = plt.subplots(2,2, figsize=(20, 12), dpi=200)
# a) Compressive strength
custom_palette = sns.color_palette(colors)
axs = ax[0,0]
# sns.boxplot(data=comp_strength_values_lithology, x='Lithology', y = 'Comp_strength', palette=custom_palette, hue = 'Lithology', ax = axs)
sns.violinplot(data=rebound_values_lithology, x='Lithology', y = 'Rebound', palette=custom_palette,  hue = 'Lithology',
               ax = axs, split = False, inner="quart", linewidth=1.5, cut=0)
# sns.stripplot(data=comp_strength_values_lithology, x='Lithology', y='Comp_strength', ax=axs, color="white",size=3, alpha=0.4, jitter=True)

for i in range(len(comp_strength_meas_per_lithology)):
    ax[0,0].text(i/10+0.05, 0.95, 'n=' + str(comp_strength_meas_per_lithology[i]), fontsize = labelsizes, ha='center', transform=ax[0,0].transAxes)
    
ax[0,0].set_ylabel('Rebound value (-)', fontsize=fontsizes, weight = 'bold')
ax[0,0].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[0,0].set_xticklabels([])
ax[0,0].set_ylim(0,85) 

# b) Compressive strength - 
custom_palette = sns.color_palette(colors)
axs = ax[0,1]
ax[0,1].bar(lithologies, smaller_10_lithology, color = colors, edgecolor = 'black')

ax[0,1].set_ylabel(r'$\mathregular{N < 10}$', fontsize=fontsizes, weight = 'bold')
ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[0,1].set_xticklabels([])
ax[0,1].set_ylim(0,) 

# c) Frac intensity
axs = ax[1,0]
# sns.boxplot(data=frac_intensity_values_lithology, x='Lithology', y = 'Intensity_whole_map_main', palette=custom_palette, hue='Lithology', ax = axs)
sns.violinplot(data=frac_intensity_values_lithology, x='Lithology', y = 'Intensity_whole_map_main', palette=custom_palette,  hue = 'Lithology',
               ax = axs, split = False, inner="quart", linewidth=1.5, cut=0)
sns.stripplot(data=frac_intensity_values_lithology, x='Lithology', y='Intensity_whole_map_main', ax=axs, color="white",size=5, alpha=0.4, jitter=True)

for i in range(len(frac_intensity_meas_per_lithology)):
    ax[1,0].text(i/10+0.05, 0.95, 'n=' + str(frac_intensity_meas_per_lithology[i]), fontsize = labelsizes, ha='center', transform=ax[1,0].transAxes)
    
ax[1,0].set_ylabel('Fracture intensity (${\mathregular{m^{-1}}}$)', fontsize=fontsizes, weight = 'bold')
ax[1,0].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[1,0].set_xticklabels([])
ax[1,0].set_ylim(0,55) 

# d) Block size 
axs = ax[1,1]
# sns.boxplot(data=block_size_values_lithology, x='Lithology', y = 'Blocksize_mean_m2_main', palette=custom_palette, hue = 'Lithology', ax = axs)
sns.violinplot(data=block_size_values_lithology, x='Lithology', y = 'Blocksize_mean_m2_main', palette=custom_palette,  hue = 'Lithology',
               ax = axs, split = False, inner="quart", linewidth=1.5, cut=0)
sns.stripplot(data=block_size_values_lithology, x='Lithology', y='Blocksize_mean_m2_main', ax=axs, color="white",size=5, alpha=0.4, jitter=True)

for i in range(len(blocksize_meas_per_lithology)):
    ax[1,1].text(i/10+0.05, 0.95, 'n=' + str(blocksize_meas_per_lithology[i]), fontsize = labelsizes, ha='center', transform=ax[1,1].transAxes)
    
ax[1,1].set_ylabel('Mean block size (m²)', fontsize=fontsizes, weight = 'bold')
ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[1,1].set_ylim(0,0.55) 
ax[1,1].set_xticklabels([])

ax[0,0].text(-0.1, 0.95, 'a)', fontsize = labelsizes, transform = ax[0,0].transAxes) 
ax[0,1].text(-0.07, 0.95, 'b)', fontsize = labelsizes, transform = ax[0,1].transAxes)
ax[1,0].text(-0.1, 0.95, 'c)', fontsize = labelsizes, transform = ax[1,0].transAxes)
ax[1,1].text(-0.07, 0.95, 'd)', fontsize = labelsizes, transform = ax[1,1].transAxes) 

fig.tight_layout()
figname = 'Lithologies_properties'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')


#%% Apply ISRM procedure

hammer_cols = data.filter(like='Hammer_value_').columns

def calculate_isrm_stats(row):
    """
    Implements ISRM procedure:
    1. Extracts top 10 readings.
    2. Calculates Mean and Standard Deviation of those 10.
    """
    # Extract values and drop NaNs
    readings = row[hammer_cols].dropna().values
    
    if len(readings) < 10:
        return pd.Series([np.nan, np.nan])
    
    # Sort descending and take the 10 highest
    top_10 = sorted(readings, reverse=True)[:10]
    
    return pd.Series([np.mean(top_10), np.std(top_10, ddof=1)]) # ddof=1 for sample std

# Apply and create two new columns at once
data[['ISRM_Mean', 'ISRM_Std']] = data.apply(calculate_isrm_stats, axis=1)

plt.plot(data['ISRM_Mean'], data['Comp_strength_mean'], ls = '', marker = "o")

#%% Apply ASTM procedure
# Identify the columns
hammer_cols = data.filter(like='Hammer_value_').columns

def calculate_astm_rebound(row):
    """
    Implements ASTM D5873 procedure:
    1. Calculate the mean of all readings.
    2. Identify readings differing by > 7 units from the mean.
    3. Discard outliers.
    4. If > 2 values are discarded, the test is invalid (ASTM strictly speaks to 10-reading sets).
    5. Return the mean of remaining values.
    """
    # Extract values and drop NaNs
    readings = row[hammer_cols].dropna().values
    
    if len(readings) == 0:
        return np.nan
    
    # Step 1: Initial Mean
    initial_mean = np.mean(readings)
    
    # Step 2: Find outliers (difference > 7 units)
    # Note: ASTM D5873 is written for sets of 10. 
    # For a 25-reading set, the logic remains: check deviation from the mean.
    valid_readings = [r for r in readings if abs(r - initial_mean) <= 7]
    discarded_count = len(readings) - len(valid_readings)
    
    # Step 3: Validity Check
    # ASTM states if more than 2 readings are discarded from a set of 10, reject it.
    # Pro-rating for 25 readings: (2/10) * 25 = 5. 
    # If using exactly 10, keep it at 2.
    limit = 2 if len(readings) <= 12 else 5 
    
    if discarded_count > limit:
        return None  # Represents an "Invalid Test"
    
    # Step 4: Final Mean of valid readings
    return np.mean(valid_readings)

data['ASTM_Rebound_Hardness'] = data.apply(calculate_astm_rebound, axis=1)


plt.plot(data['ASTM_Rebound_Hardness'], data['Comp_strength_mean'], ls = '', marker = "o")


#%% Figure article ISRM 
data_plot = data.dropna(subset=['ISRM_Mean']).copy()

fig, ax = plt.subplots(2,2, figsize=(20, 12), dpi=200)
# a) Compressive strength
custom_palette = sns.color_palette(colors)
palette_dict = dict(zip(lithologies, colors))
axs = ax[0,0]
# sns.boxplot(data=comp_strength_values_lithology, x='Lithology', y = 'Comp_strength', palette=custom_palette, hue = 'Lithology', ax = axs)
sns.violinplot(data=data_plot, x='Lithology', y='ISRM_Mean', order=lithologies, palette=palette_dict,  hue = 'Lithology', legend=False,
               ax = axs, split = False, inner="quart", linewidth=1.5, cut=0)
# sns.stripplot(data=comp_strength_values_lithology, x='Lithology', y='Comp_strength', ax=axs, color="white",size=3, alpha=0.4, jitter=True)

for i in range(len(comp_strength_meas_per_lithology)):
    ax[0,0].text(i/10+0.05, 0.95, 'n=' + str(comp_strength_meas_per_lithology[i]), fontsize = labelsizes, ha='center', transform=ax[0,0].transAxes)
    
ax[0,0].set_ylabel('Rebound value (-)', fontsize=fontsizes, weight = 'bold')
ax[0,0].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[0,0].set_xticklabels([])
ax[0,0].set_ylim(0,85) 

# b) Compressive strength - 
custom_palette = sns.color_palette(colors)
axs = ax[0,1]
ax[0,1].bar(lithologies, smaller_10_lithology, color = colors, edgecolor = 'black')

ax[0,1].set_ylabel(r'$\mathregular{N < 10}$', fontsize=fontsizes, weight = 'bold')
ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[0,1].set_xticklabels([])
ax[0,1].set_ylim(0,) 

# c) Frac intensity
axs = ax[1,0]
# sns.boxplot(data=frac_intensity_values_lithology, x='Lithology', y = 'Intensity_whole_map_main', palette=custom_palette, hue='Lithology', ax = axs)
sns.violinplot(data=frac_intensity_values_lithology, x='Lithology', y = 'Intensity_whole_map_main', palette=custom_palette,  hue = 'Lithology',
               ax = axs, split = False, inner="quart", linewidth=1.5, cut=0)
sns.stripplot(data=frac_intensity_values_lithology, x='Lithology', y='Intensity_whole_map_main', ax=axs, color="white",size=5, alpha=0.4, jitter=True)

for i in range(len(frac_intensity_meas_per_lithology)):
    ax[1,0].text(i/10+0.05, 0.95, 'n=' + str(frac_intensity_meas_per_lithology[i]), fontsize = labelsizes, ha='center', transform=ax[1,0].transAxes)
    
ax[1,0].set_ylabel('Fracture intensity (${\mathregular{m^{-1}}}$)', fontsize=fontsizes, weight = 'bold')
ax[1,0].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[1,0].set_xticklabels([])
ax[1,0].set_ylim(0,55) 

# d) Block size 
axs = ax[1,1]
# sns.boxplot(data=block_size_values_lithology, x='Lithology', y = 'Blocksize_mean_m2_main', palette=custom_palette, hue = 'Lithology', ax = axs)
sns.violinplot(data=block_size_values_lithology, x='Lithology', y = 'Blocksize_mean_m2_main', palette=custom_palette,  hue = 'Lithology',
               ax = axs, split = False, inner="quart", linewidth=1.5, cut=0)
sns.stripplot(data=block_size_values_lithology, x='Lithology', y='Blocksize_mean_m2_main', ax=axs, color="white",size=5, alpha=0.4, jitter=True)

for i in range(len(blocksize_meas_per_lithology)):
    ax[1,1].text(i/10+0.05, 0.95, 'n=' + str(blocksize_meas_per_lithology[i]), fontsize = labelsizes, ha='center', transform=ax[1,1].transAxes)
    
ax[1,1].set_ylabel('Mean block size (m²)', fontsize=fontsizes, weight = 'bold')
ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[1,1].set_ylim(0,0.55) 
ax[1,1].set_xticklabels([])

ax[0,0].text(-0.1, 0.95, 'a)', fontsize = labelsizes, transform = ax[0,0].transAxes) 
ax[0,1].text(-0.07, 0.95, 'b)', fontsize = labelsizes, transform = ax[0,1].transAxes)
ax[1,0].text(-0.1, 0.95, 'c)', fontsize = labelsizes, transform = ax[1,0].transAxes)
ax[1,1].text(-0.07, 0.95, 'd)', fontsize = labelsizes, transform = ax[1,1].transAxes) 

fig.tight_layout()
figname = 'Lithologies_properties_ISRM'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')


#%% Figure article ISRM rock strengths - Supporting info
# Calculate tensile strength using Jamshidi et al., 2018
data_plot['Tensile_strength_Pa'] = 0.0347*10**6*data_plot['ISRM_Mean']**1.3783
data_plot['Tensile_strength_Pa_std'] = 0.0347*10**6*data_plot['ISRM_Std']**1.3783
data_plot['Tensile_strength_MPa'] = data_plot['Tensile_strength_Pa']/10**6
data_plot['Tensile_strength_MPa_std'] = data_plot['Tensile_strength_Pa_std']/10**6
data_plot['Compressive_strength_MPa'] = 1.624*data_plot['ISRM_Mean']- 17.23 # assuming vertical
data_plot['Compressive_strength_MPa_std'] = (1.624*data_plot['ISRM_Std']- 17.23)*(-1)

fig, ax = plt.subplots(1,2, figsize=(14, 7), dpi=200)
# a) Compressive strength
custom_palette = sns.color_palette(colors)
palette_dict = dict(zip(lithologies, colors))
axs = ax[0]
sns.violinplot(data=data_plot, x='Lithology', y='Compressive_strength_MPa', order=lithologies, palette=palette_dict,  hue = 'Lithology', legend=False,
               ax = axs, split = False, inner="quart", linewidth=1.5, cut=0)
sns.stripplot(data=data_plot, x='Lithology', y='Compressive_strength_MPa', ax=axs, color="white",size=3, alpha=0.4, jitter=True)

ax[0].set_xlabel('', fontsize=18, weight = 'bold') 
ax[0].set_ylabel('Compressive strength (MPa)', fontsize=fontsizes, weight = 'bold')
ax[0].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[0].set_xticklabels(lithologies_label, rotation=45, ha='right')
ax[0].set_ylim(0,) 

# b) Tensile strength 
axs = ax[1]
sns.violinplot(data=data_plot, x='Lithology', y='Tensile_strength_MPa', order=lithologies, palette=palette_dict,  hue = 'Lithology', legend=False,
               ax = axs, split = False, inner="quart", linewidth=1.5, cut=0)
sns.stripplot(data=data_plot, x='Lithology', y='Tensile_strength_MPa', ax=axs, color="white",size=3, alpha=0.4, jitter=True)

ax[1].set_ylabel('Tensile strength (MPa)', fontsize=fontsizes, weight = 'bold')
ax[1].tick_params(axis = 'both', which = 'major', labelsize = labelsizes)
ax[1].set_xticklabels(lithologies_label, rotation=45, ha='right')
ax[1].set_ylim(0,) 
ax[1].set_xlabel('', fontsize=18, weight = 'bold')   
ax[0].text(-0.15, 1.02, 'a)', fontsize = labelsizes, transform = ax[0].transAxes) 
ax[1].text(-0.15, 1.02, 'b)', fontsize = labelsizes, transform = ax[1].transAxes)

fig.tight_layout()
figname = 'Lithologies_strengths_ISRM_SI'
fig.savefig(outpath_figures+ '\\' + figname + '.png', dpi=300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname + '.eps', dpi=300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname + '.pdf', dpi=300, bbox_inches='tight')

#%% Average and output based on unit lithologies close to outcrops

aa = [data[data['Lithology']==lithologies[i]].mean(numeric_only=True).transpose()
      for i in range(len(lithologies))]
fracture_intensity_std = [data['Intensity_whole_map_main'][data['Lithology']==lithologies[i]].std()
      for i in range(len(lithologies))]
fracture_density_std = [data['Density_whole_map_main'][data['Lithology']==lithologies[i]].std()
      for i in range(len(lithologies))]

gg = [grain_size[grain_size['Lithology_ID']==lithologies_ID[i]].mean(numeric_only=True).transpose()
      for i in range(len(lithologies))]
grain_size_averaged = pd.DataFrame(gg)

data_for_comparison_flume_meas = pd.DataFrame(aa)
data_for_comparison_flume_meas['Intensity_whole_map_main_std'] = fracture_intensity_std
data_for_comparison_flume_meas['Density_whole_map_main_std'] = fracture_density_std
data_for_comparison_flume_meas['Lithology']=lithologies
data_for_comparison_flume_meas['Lithology_ID'] = np.arange(1,11,1)
data_for_comparison_flume_meas['Mineral_D50_mum']=grain_size_averaged['Mineral_D50']
data_for_comparison_flume_meas['Mineral_weighted_std_mum']=grain_size_averaged['Mineral_weighted_std']

data_for_comparison_flume_meas.to_csv(r'filepath\Data_for_comparison_flume_meas.csv')
data.to_csv(r'filepath\All_field_data_article_outcrops.csv', index=False)
grain_size.to_csv(r'filepath\Mineral_grain_size_D50.csv', index=False)
grain_size_averaged.to_csv(r'filepath\Mineral_grain_size_lithology_averaged.csv', index=False)















