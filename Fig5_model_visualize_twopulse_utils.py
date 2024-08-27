import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def plot_broadband(data, data_one_pulse, data_two_pulse, trials_lbls, start, tempCond, duration, n_img, color_trial, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 15
    fontsize_tick           = 10

    # initiate figure
    fig, axs = plt.subplots(1, len(tempCond), figsize=(17, 2))
    sns.despine(offset=10)

    # start
    start_plot = 2

    # select layer
    iL = 0

    # PLOT AVERAGE
    for iT, _ in enumerate(trials_lbls):

        for iC in range(len(tempCond)):

            # select data
            data_current = data[iT, iL, iC, :, start_plot:] - data[iT, iL, iC, :, -1].reshape(-1, 1)

            # normalize data
            for iImg in range(n_img):

                axs[iC].plot(np.arange(data_current.shape[1])-start_plot, data_current[iImg, :], color=color_trial[iT], lw=0.2, alpha=0.2)
            
            # compute
            data_mean = np.mean(data_current, 0)

            # visualize
            axs[iC].plot(np.arange(data_current.shape[1])-start_plot, data_mean, color=color_trial[iT], markersize=1, marker='o', lw=1)

            # adjust axes
            axs[iC].tick_params(axis='both', labelsize=fontsize_tick)
            axs[iC].axhline(0, lw=0.5, color='grey')
            axs[iC].axvline(0, lw=0.5, color='grey')   
            axs[iC].spines['top'].set_visible(False)
            axs[iC].spines['right'].set_visible(False)
            if (iC > 0):
                axs[iC].spines['left'].set_visible(False)
                axs[iC].spines['right'].set_visible(False)
                axs[iC].set_yticks([])
            if iT == 0:
                axs[iC].axvspan(0, duration, facecolor='lightgrey', edgecolor=None, alpha=0.5)
                axs[iC].axvspan(tempCond[iC]+duration, tempCond[iC]+2*duration, facecolor='lightgrey', edgecolor=None, alpha=0.5)

    # save
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(root + '/visualization/Fig5A_model', dpi=600)
    plt.savefig(root + '/visualization/Fig5A_model.svg')
    plt.close()

def plot_dynamics(dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 17
    fontsize_tick           = 12

    # initiate figure
    _, axs = plt.subplots(1, n_layer, figsize=(12, 3), sharey=True)
    sns.despine(offset=10)

    linestyle = ['solid', 'dashdot']
    color_fit = ['lightgrey', 'darkgrey']
    lw = 0.75

    offset_trial    = [-0.5, 0.5]
    for iL in range(n_layer):

        for iT, trial in enumerate(trials):

            # select data
            data_current = metric[iT, iL, :, :]

            # compute spread
            data_mean = np.mean(data_current, 1)
            data_sem = np.std(data_current, 1)/math.sqrt(n_img)
            
            # plot
            axs[iL].scatter(tempCond+offset_trial[iT], data_mean, facecolor=color_trial[iT], edgecolor='white', s=80, zorder=1)

            # plot spread
            for iC in range(len(tempCond)):
                axs[iL].plot([tempCond[iC]+offset_trial[iT], tempCond[iC]+offset_trial[iT]], [data_mean[iC] - data_sem[iC], data_mean[iC] + data_sem[iC]], color=color_trial[iT], zorder=-10)

            # select data
            data_current = dynamics_log[iT, iL, :, :]

            # compute
            data_mean = np.mean(data_current, 0)

            # plot
            axs[iL].plot(t1_plot, data_mean, color=color_trial[iT], lw=lw, linestyle=linestyle[1], zorder=-10)

        # adjust axes
        axs[iL].tick_params(axis='both', labelsize=fontsize_tick)  
        axs[iL].spines['top'].set_visible(False)
        axs[iL].spines['right'].set_visible(False)
        axs[iL].set_xticks(tempCond)
        axs[iL].set_xticklabels(tempCond, rotation=45)
        axs[iL].axhline(1, lw=2, linestyle='--', zorder=-10, color='darkgrey')
        axs[iL].set_ylim(0.4, 1.2)
        # if iL == 0:    
        #     axs[iL].set_ylabel('Response magnitude', fontsize=fontsize_label)
        # else:
        #     axs[iL].set_yticklabels([])

    # save fig
    plt.tight_layout()
    plt.savefig(root + '/visualization/Fig5D_model', dpi=600)
    plt.savefig(root + '/visualization/Fig5D_model.svg')
    plt.close()

def plot_dynamics_metric(dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 17
    fontsize_tick           = 12

    # initiate figure
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    sns.despine(offset=10)

    linestyle = ['solid', 'dashdot']
    color_fit = ['lightgrey', 'darkgrey']
    lw = 2

    offset_L        = [0, 4, 8, 12]
    offset_trial    = [-0.5, 0.5]
    for iL in range(n_layer):
        for iT, trial in enumerate(trials):

            #### PLOT METRIC
            ax.set_xticks([offset_L[0], offset_L[1], offset_L[2]])

            # select data
            data_current = adaptation_avg[iT, iL, :]

            # compute
            data_mean = np.median(data_current)
            data_sem = np.std(data_current)/math.sqrt(n_img)

            # plot
            ax.scatter(offset_L[iL]+offset_trial[iT], data_mean, color=color_trial[iT], edgecolor='white', s=100)

            # plot spread (sem)
            ax.plot([offset_L[iL]+offset_trial[iT], offset_L[iL]+offset_trial[iT]], [data_mean - data_sem, data_mean], color=color_trial[iT], zorder=-10)

            # plot individual electrodes
            sns.stripplot(x=np.ones(n_img)*(offset_L[iL]+offset_trial[iT]), y=adaptation_avg[iT, iL, :], jitter=True, ax=ax, color=color_trial[iT], size=4, alpha=0.2, native_scale=True)

    # adjust axes
    ax.tick_params(axis='both', labelsize=fontsize_tick)  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([offset_L[0], offset_L[1], offset_L[2], offset_L[3]])
    ax.set_xticklabels(['E1', 'E2', 'E3', 'E4'])
    # ax.set_ylim(0.3, 1)
    ax.axhline(1, lw=2, linestyle='--', zorder=-10, color='darkgrey') 
    ax.set_ylabel('Avg. adaptation', fontsize=fontsize_label)

    # save fig
    plt.tight_layout()
    plt.savefig(root + '/visualization/Fig5E_model', dpi=600)
    plt.savefig(root + '/visualization/Fig5E_model.svg')
    plt.close()

def stats_trial(adaptation_avg, n_layer):

    alpha = 0.05

    for iL in range(n_layer):

        # print progress
        print('Layer', iL+1)

        # ttest
        sample1 = adaptation_avg[0, iL, :]
        sample2 = adaptation_avg[1, iL, :]
        p = stats.ttest_rel(sample1, sample2)[1]
        if p < alpha:
            print('repetition vs. alternation', p, ' SIGNIFICANT')
        else:
            print('repetition vs. alternation', p)

        print('\n')