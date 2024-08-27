import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import pandas as pd
import math
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.utils import resample
from scipy import stats
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter1d

def plot_broadband(data, bootstrapped, tempCond, trials_lbls, visual_area, n_electrodes, n_electrodes_per_area, t, timepoints_twopulse, color, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 15
    fontsize_tick           = 10

    # initiate figure
    fig, axs = plt.subplots(1, len(tempCond), figsize=(17, 2))
    sns.despine(offset=10)

    # plot settings
    lw = 2
    t_end = 700

    # select visual area
    iVA = 0

    # visualize
    for iT, trial in enumerate(trials_lbls):
        for iC in range(len(tempCond)):
                
            # plot data - normalized
            if bootstrapped:
                data_current = data[iVA, iT, iC, :, :]/np.max(data[iT, iVA, -1, :, :], 1).reshape(-1, 1)
            else:
                data_current = data[iVA][iT, iC, :, :]/np.max(data[iVA][iT, -1, :, :], 1).reshape(-1, 1)

            data_mean = gaussian_filter1d(np.mean(data_current[:, :t_end], 0), 20)
            data_sem = np.std(data_current[:, :t_end], 0)/math.sqrt(n_electrodes_per_area[iVA])

            axs[iC].plot(t[:t_end], data_mean, color=color[iT], lw=lw)
            axs[iC].fill_between(t[:t_end], data_mean - data_sem, data_mean + data_sem, alpha=0.2, edgecolor=None, facecolor=color[iT])

        # adjust axes
        for iC in range(len(tempCond)):
            axs[iC].tick_params(axis='both', labelsize=fontsize_tick)
            axs[iC].axhline(0, lw=0.5, color='grey', zorder=-10)  
            axs[iC].axvline(0, lw=0.5, color='grey', zorder=-10)
            axs[iC].spines['top'].set_visible(False)
            axs[iC].spines['right'].set_visible(False)
            axs[iC].set_ylim(-0.1, 1.1)
            axs[iC].axvspan(t[timepoints_twopulse[iC, 0]], t[timepoints_twopulse[iC, 1]], facecolor='lightgrey', edgecolor=None, alpha=0.5)
            axs[iC].axvspan(t[timepoints_twopulse[iC, 2]], t[timepoints_twopulse[iC, 3]], facecolor='lightgrey', edgecolor=None, alpha=0.5)
            if iC > 0:
                axs[iC].spines['left'].set_visible(False)
                axs[iC].set_yticks([])
            # axs[iC].set_xlabel('Time (s)', fontsize=fontsize_label)

    # save
    # axs[1, 0].legend(frameon=False, fontsize=fontsize_legend)
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig5A_data', dpi=600)
    plt.savefig(root + 'visualization/Fig5A_data.svg')
    plt.close()

def plot_dynamics(metric, adaptation_avg, dynamics_lin, dynamics_log, visual_area, trial_lbls, bootstrapped, computation, tempCond, CI_low, CI_high, color_area, color_trial, n_electrodes_per_area, t1_plot, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 17
    fontsize_tick           = 12

    # join dynamics
    dynamics = [dynamics_lin, dynamics_log]

    # initiate figure
    _, axs = plt.subplots(1, len(visual_area), figsize=(12, 3))
    sns.despine(offset=10)

    linestyle = ['solid', 'dashdot']
    color_fit = ['lightgrey', 'darkgrey']
    lw = 2

    offset              = [-5, 0, 5]
    for iVA, VA in enumerate(visual_area):

        for iT, trial in enumerate(trial_lbls):

            # select data
            if bootstrapped:

                # select data
                data_current = metric[iVA, iT, :, :]

                # compute spread
                data_mean = np.mean(data_current, 1)

            else:

                # select data
                data_current = metric[iVA][iT, :, :]

                # compute spread
                data_mean = np.mean(data_current, 1)
                data_sem = np.std(data_current, 1)/math.sqrt(n_electrodes_per_area[iVA])
            
            # plot
            axs[iVA].scatter(tempCond+offset[iVA], data_mean, facecolor=color_trial[iT], edgecolor='white', s=80, zorder=1)

            # plot spread
            for iC in range(len(tempCond)):
                if bootstrapped:
                    data_CI = np.nanpercentile(data_current[iC, :], [CI_low, CI_high])
                    axs[iVA].plot([tempCond[iC]+offset[iVA], tempCond[iC]+offset[iVA]], [data_CI[0], data_CI[1]], color=color_trial[iT], zorder=-10)
                else:
                    axs[iVA].plot([tempCond[iC]+offset[iVA], tempCond[iC]+offset[iVA]], [data_mean[iC] - data_sem[iC], data_mean[iC] + data_sem[iC]], color=color_trial[iT], zorder=-10)

            # plot dynamics fit
            for iComp in range(1, len(computation)):

                # select data
                if bootstrapped:
                    data_current = dynamics[iComp][iVA, iT, :, :]
                else:
                    data_current = dynamics[iComp][iVA][iT, :, :]

                # compute
                data_mean = np.mean(data_current, 0)

                # plot
                axs[iVA].plot(t1_plot+offset[iVA], data_mean, color=color_trial[iT], lw=0.75, linestyle=linestyle[iComp], zorder=-10)

        # adjust axes
        axs[iVA].tick_params(axis='both', labelsize=fontsize_tick)  
        axs[iVA].spines['top'].set_visible(False)
        axs[iVA].spines['right'].set_visible(False)
        axs[iVA].set_xticks(tempCond)
        axs[iVA].set_xticklabels(tempCond, rotation=45)
        axs[iVA].set_ylim(0.3, 1.1)
        axs[iVA].axhline(1, lw=2, linestyle='--', zorder=-10, color='darkgrey')
        # if iVA == 1:
        #     axs[iVA].set_xlabel('Duration (s)', fontsize=fontsize_label)
        if iVA == 0:    
            axs[iVA].set_ylabel('Response magnitude', fontsize=fontsize_label)
        else:
            axs[iVA].set_yticklabels([])

    # save fig
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig5B_data', dpi=600)
    plt.savefig(root + 'visualization/Fig5B_data.svg')
    plt.close()

def plot_dynamics_metric(metric, adaptation_avg, dynamics_lin, dynamics_log, visual_area, trial_lbls, bootstrapped, computation, tempCond, CI_low, CI_high, color_area, color_trial, n_electrodes_per_area, t1_plot, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 17
    fontsize_tick           = 12

    # join dynamics
    dynamics = [dynamics_lin, dynamics_log]

    # initiate figure
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    sns.despine(offset=10)

    linestyle = ['solid', 'dashdot']
    color_fit = ['lightgrey', 'darkgrey']
    lw = 2

    offset_VA       = [0, 4, 8]
    offset_trial    = [-0.5, 0.5]

    offset              = [-5, 0, 5]
    for iVA, VA in enumerate(visual_area):
        for iT, trial in enumerate(trial_lbls):

            #### PLOT METRIC
            ax.set_xticks([offset_VA[0], offset_VA[1], offset_VA[2]])
            if bootstrapped:

                # select data
                data_current = adaptation_avg[iVA, iT, :]

                # compute
                data_mean = np.mean(data_current)
                data_CI = np.nanpercentile(data_current, [CI_low, CI_high])

            else:

                # select data
                data_current = adaptation_avg[iVA][iT, :]

                # compute
                data_mean = np.median(data_current)
                data_sem = np.std(data_current)/math.sqrt(n_electrodes_per_area[iVA])

            # plot
            ax.scatter(offset_VA[iVA]+offset_trial[iT], data_mean, color=color_trial[iT], edgecolor='white', s=100)

            # plot spread
            if bootstrapped:
                ax.plot([offset_VA[iVA]+offset_trial[iT], offset_VA[iVA]+offset_trial[iT]], [data_CI[0], data_CI[1]], color=color_trial[iT], zorder=-10)
            else:

                # plot spread (sem)
                ax.plot([offset_VA[iVA]+offset_trial[iT], offset_VA[iVA]+offset_trial[iT]], [data_mean - data_sem, data_mean], color=color_trial[iT], zorder=-10)

                # plot individual electrodes
                sns.stripplot(x=np.ones(n_electrodes_per_area[iVA])*(offset_VA[iVA]+offset_trial[iT]), y=adaptation_avg[iVA][iT, :], jitter=True, ax=ax, color=color_trial[iT], size=6, alpha=0.2, native_scale=True)

    # adjust axes
    ax.tick_params(axis='both', labelsize=fontsize_tick)  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([offset_VA[0], offset_VA[1], offset_VA[2]])
    ax.set_xticklabels(visual_area)
    ax.set_ylim(0.3, 1.1)
    ax.axhline(1, lw=2, linestyle='--', zorder=-10, color='darkgrey') 
    ax.set_ylabel('Avg. adaptation', fontsize=fontsize_label)

    # save fig
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig5C_data', dpi=600)
    plt.savefig(root + 'visualization/Fig5C_data.svg')
    plt.close()

def stats_trial(adaptation_avg, visual_area):

    alpha = 0.05

    for iVA in range(len(visual_area)):

        # print progress
        print(iVA)

        # ttest
        sample1 = adaptation_avg[iVA][0, :]
        sample2 = adaptation_avg[iVA][1, :]
        p = stats.ttest_rel(sample1, sample2)
        if p[1] < alpha:
            print('repetition vs. alternation', p, ' SIGNIFICANT')
        else:
            print('repetition vs. alternation', p)

        print('\n')