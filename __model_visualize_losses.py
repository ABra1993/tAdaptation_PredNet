import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from neural_data_visualize_utils import *
from __model_visualize_onepulse_utils import *

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def subsample(data, sample_size):
    # Desired downsampled length
    sample_size = 150

    # Calculate the step size for downsampling
    step_size = len(data) // sample_size

    # Downsampled sequence
    downsampled_sequence = data[::step_size]

    # Ensure the downsampled sequence has the exact desired length
    if len(downsampled_sequence) > sample_size:
        downsampled_sequence = downsampled_sequence[:sample_size]

    return downsampled_sequence

# set root
root            = '/prednet_Brands2024_git/'
data_save       = '/prednet_Brands2024_git/data/stimuli/img_statistics/' # for plotting cross-correlation

# datasets
datasets            = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']
datasets_lbls       = ['KITTI', 'Amsterdam', 'Venice', 'wildlife']
color               = ['#F6C141', '#4EB265', '#5289C7', '#DC050C']

# training settings
epochs      = 150
n_samples   = 500

#####################################################################################
#####################################################################################
#####################################################################################

input_motion        = ['dynamic', 'static']

# network init.
init = 5

# save average losses
losses_avg = np.zeros((len(datasets), len(input_motion), init))

# plot losses
fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw=dict(width_ratios=[2, 1]))
sns.despine(offset=10)

# fontsizes 
fontsize_title          = 20
fontsize_legend         = 12
fontsize_label          = 18
fontsize_tick           = 15

lw = 2
alpha = [0.3, 1 ]

plt.subplots_adjust(wspace=4)

for iD, dataset in enumerate(datasets):

    # print('Current dataset: ', dataset)

    losses = np.zeros((init, len(input_motion), int(epochs*n_samples)))

    for i in range(2):

        # load losses
        for iInit in range(init):
            if i == 0:
                losses[iInit, 0, :] = np.loadtxt(root + 'losses/dataset/static_' + dataset + str(iInit+1) + '_log_t.txt')
            elif i == 1:
                losses[iInit, 1, :] = np.loadtxt(root + 'losses/dataset/' + dataset + str(iInit+1) + '_log_t.txt')

        # compute data
        data_mean = gaussian_filter1d(np.mean(losses[:, i, :], 0), 5)
        data_sem = gaussian_filter1d(np.std(losses[:, i, :], 0), 5)/math.sqrt(init)
        # data_mean = np.mean(losses[:, i, :], 0)
        # data_sem = np.std(losses[:, i, :], 0)/math.sqrt(init)
        # print(data_sem)

        # subsample
        data_mean = subsample(data_mean, 500)
        data_sem = subsample(data_sem, 500)

        # visualize
        if i == 1:
            axs[0].plot(data_mean, color=color[iD], alpha=alpha[i], lw=lw)
            axs[0].fill_between(np.arange(150), data_mean - data_sem, data_mean + data_sem, edgecolor='white', color=color[iD], alpha=0.2)
        
        if i == 0:
            print(dataset, ': ', np.mean(data_mean))
            print('    std: ', np.mean(data_sem))

    # compute difference in performance
    for iInit in range(init):
        # losses_avg[iD, 0, iInit] = np.mean(subsample(gaussian_filter1d(losses[iInit, 0, :], 10), 500))
        # losses_avg[iD, 1, iInit] = np.mean(subsample(gaussian_filter1d(losses[iInit, 1, :], 10), 500))
        losses_avg[iD, 0, iInit] = np.mean(subsample(losses[iInit, 0, :], 500))
        losses_avg[iD, 1, iInit] = np.mean(subsample(losses[iInit, 1, :], 500))

# statistics
result = f_oneway(losses_avg[0, 0, :], losses_avg[1, 0, :], losses_avg[2, 0, :], losses_avg[3, 0, :])
print(result)


result = f_oneway(losses_avg[0, 1, :], losses_avg[1, 1, :], losses_avg[2, 1, :], losses_avg[3, 1, :])
print(result)

# adjust axes
# axs[0].set_yscale('log')
axs[0].tick_params(axis='both', labelsize=fontsize_tick)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_ylim(0, 0.40)
# axs[0].set_xlabel('Epoch', fontsize=fontsize_label)

# axs[0].set_ylabel(r'Loss$_{train}$', fontsize=fontsize_label)
# axs[0].set_ylim(0, 0.15)

# compute mean
data_mean = np.mean(losses_avg, 1)

# compute difference
offset_iM = [-0.5, 0.5]
offset_iD = [0, 3, 6, 9]
for iD, dataset in enumerate(datasets_lbls):
    for iM, motion in enumerate(input_motion):

        # print('Current dataset: ', dataset)

        # select data
        data_current = losses_avg[iD, iM, :]

        # compute spread
        data_mean   = np.mean(data_current)
        data_sem    = np.std(data_current)/math.sqrt(init)

        # visualize
        axs[1].bar(offset_iD[iD]+offset_iM[iM], data_mean, color=color[iD], alpha=alpha[iM])
        axs[1].plot([offset_iD[iD]+offset_iM[iM], offset_iD[iD]+offset_iM[iM]], [data_mean - data_sem, data_mean + data_sem], color='darkgrey')

# adjust axes
axs[1].tick_params(axis='both', labelsize=fontsize_tick)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
# axs[1].set_ylabel('Avg. loss', fontsize=fontsize_label)
axs[1].set_xticks(offset_iD)
axs[1].set_xticklabels(datasets_lbls, rotation=30, ha='right')

# create correlation
axins = inset_axes(axs[0], width="20%", height="40%")

for iD, dataset in enumerate(datasets):

        print('Current dataset: ', dataset)

        # select data
        data_current = np.load(data_save + 'crossCorr_' + dataset + '.npy')

        # compute spread
        data_mean   = np.mean(data_current)

        # visualize
        axins.bar(iD, data_mean, color=color[iD])

# adjust axes
axins.tick_params(axis='both', labelsize=fontsize_tick)
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)
axins.set_xticks([])
axins.set_yticks([])
axins.set_ylabel(r'$\mathit{AutoCorr}$', fontsize=10)


# save figure
plt.tight_layout()
# plt.legend(frameon=False)
plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Kirubeswaran2023/datasets/losses/losses_dataset', dpi=300)
plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Kirubeswaran2023/datasets/losses/losses_dataset.svg')

#####################################################################################
#####################################################################################
#####################################################################################

# select dataset
datasets_manipulation = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']

# network init.
init = 3

# fontsizes 
fontsize_title          = 20
fontsize_legend         = 15
fontsize_label          = 18
fontsize_tick           = 16

lw = 2
alpha = [0.3, 1]

# maximum values
ylim = [[0, 0.6], [0, 0.6], [0, 0.6], [0, 0.2]]

for iDM, dataset_manipulation in enumerate(datasets_manipulation):
    for i in range(2):

        if i == 0:
            continue

        # set dataset
        if i == 0:
            title               = 'fps'
            manipulations       = ['_fps3', '_fps6', '_fps12']
            lbls                = ['3', '6', '12']
            color               = ['khaki', 'sandybrown', 'tomato']
        elif i == 1:
            title               = 'loss'
            manipulations       = ['_Lnull', '_Lall']
            lbls                = [r'L$_{0}$', r'L$_{all}$']
            color               = ['#125A56', '#238F9D']

        # initialize figure
        if i == 0:
            fig = plt.figure(figsize=(4, 2.5))
        else:
            fig = plt.figure(figsize=(4, 2.5))
        axs = plt.gca()

        # store losses
        losses = np.zeros((init, int(epochs*n_samples)))

        for iM, manip in enumerate(manipulations):

            # load losses
            for iInit in range(init):

                # extract
                losses[iInit, :] = np.loadtxt(root + 'losses/' + title + '/' + dataset_manipulation + manip + '_' + str(iInit+1) + '_log_t.txt')

            # compute data
            data_mean   = gaussian_filter1d(np.mean(losses, 0), 50)
            data_sem    = gaussian_filter1d(np.std(losses, 0), 50)/math.sqrt(init)
            # print(data_sem)

            # subsample
            data_mean   = subsample(data_mean, 500)
            print(dataset_manipulation, ', loss: ', np.mean(data_mean))
            data_sem    = subsample(data_sem, 500)
            print('-- dev: ', np.mean(data_sem))

            # # visualize
            axs.plot(data_mean, color=color[iM], lw=lw, label=lbls[iM])
            axs.fill_between(np.arange(150), data_mean - data_sem, data_mean + data_sem, edgecolor=color[iM], color=color[iM], alpha=0.2)

        # set legend
        axs.tick_params(axis='both', labelsize=fontsize_tick)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.set_ylim(ylim[iDM][0], ylim[iDM][1])
        # if dataset_manipulation == 'WT_WL':
        #     axs.set_yticks([0.02, 0.04, 0.06, 0.08])
        plt.tight_layout()

        # axs.set_xlabel('Epoch', fontsize=fontsize_label)
        # axs.set_ylabel('Loss', fontsize=fontsize_label)
        # axs.legend(frameon=False, title=title)

        # save figure
        plt.tight_layout()
        if i == 0:
            plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Kirubeswaran2023/datasets/losses/losses_fps_' + dataset_manipulation, dpi=300)
            plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Kirubeswaran2023/datasets/losses/losses_fps_'  + dataset_manipulation + '.svg')
        else:
            plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Kirubeswaran2023/datasets/losses/losses_loss_' + dataset_manipulation, dpi=300)
            plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Kirubeswaran2023/datasets/losses/losses_loss_' + dataset_manipulation + '.svg')
