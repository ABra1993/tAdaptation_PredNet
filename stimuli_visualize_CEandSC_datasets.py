import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from neural_data_visualize_utils import *
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# import values
n_img = 500

# datasets
datasets            = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']
datasets_lbls       = ['KITTI', 'Amsterdam', 'Venice', 'Wild life']
color               = ['#F6C141', '#4EB265', '#5289C7', '#DC050C']

# import
CEandSC_values = np.zeros((len(datasets), n_img, 2))
for iD, dataset in enumerate(datasets):
    temp = np.load('/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/' + dataset + '.npy')
    CEandSC_values[iD, :, :] = temp[:n_img, :]

# initiate figure
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
sns.despine(offset=10)

# fontsizes 
fontsize_title          = 20
fontsize_legend         = 12
fontsize_label          = 18
fontsize_tick           = 15

# adjust axes
ax.tick_params(axis='both', labelsize=fontsize_tick)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('CE', fontsize=fontsize_label)
ax.set_xlabel('SC', fontsize=fontsize_label)

for iD, dataset in enumerate(datasets_lbls):

    # plot
    # plt.scatter(CEandSC_values[iD, :n_img, 1], CEandSC_values[iD, :n_img, 0], color=color[iD], alpha=0.6, s=20, edgecolor='white', zorder=-10)
    plt.scatter(CEandSC_values[iD, :n_img, 1], CEandSC_values[iD, :n_img, 0], facecolor=color[iD], marker='.', s=30, zorder=-10, alpha=0.25, edgecolors='none')
    
    # compute values
    data_mean_ce = np.mean(CEandSC_values[iD, :n_img, 0])
    data_mean_sc = np.mean(CEandSC_values[iD, :n_img, 1])

    data_std_ce = np.std(CEandSC_values[iD, :n_img, 0])/math.sqrt(n_img)
    data_std_sc = np.std(CEandSC_values[iD, :n_img, 1])/math.sqrt(n_img)

    # visualize mean
    plt.scatter(data_mean_sc, data_mean_ce, color=color[iD], label=dataset, s=100, edgecolor='white', zorder=1)

    # visualize spread
    plt.plot([data_mean_sc - data_std_sc, data_mean_sc + data_std_sc], [data_mean_ce, data_mean_ce], color=color[iD], zorder=-10)
    plt.plot([data_mean_sc, data_mean_sc], [data_mean_ce - data_std_ce, data_mean_ce + data_std_ce],  color=color[iD], zorder=-10)

# adjust axes
ax.legend(frameon=False)
ax.set_xlabel('SC')
ax.set_ylabel('CE')
ax.set_xlim(0.4, 1.9)
ax.set_ylim(-0.001, 0.008)

# save figure
plt.tight_layout()
plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/stimuli/img_statistics/CEandSCvalues_datasets', dpi=300)
plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/stimuli/img_statistics/CEandSCvalues_datasets.svg')