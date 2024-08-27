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
from scipy.stats import tukey_hsd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter1d

def plot_broadband(data_all, data, bootstrapped, tempCond, visual_area, n_electrodes, n_electrodes_per_area, t, timepoints_onepulse, range_trans, range_sust, color, root):

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

    # PLOT AVERAGE
    for iC in range(len(tempCond)):
        for iVA, VA in enumerate(visual_area):
            
            # plot data - normalized
            if bootstrapped:
                data_current = data[iVA, iC, :, :]/np.max(data[iVA, -1, :, :], 1).reshape(-1, 1)
            else:
                data_current = data[iVA][iC, :, :]/np.max(data[iVA][-1, :, :], 1).reshape(-1, 1)

            data_mean = gaussian_filter1d(np.mean(data_current[:, :t_end], 0), 20)
            data_sem = np.std(data_current[:, :t_end], 0)/math.sqrt(n_electrodes_per_area[iVA])

            axs[iC].plot(t[:t_end], data_mean, color=color[iVA], lw=lw, label=VA)
            axs[iC].fill_between(t[:t_end], data_mean - data_sem, data_mean + data_sem, alpha=0.2, edgecolor=None, facecolor=color[iVA])

    # adjust axes
    for iC in range(len(tempCond)):
        axs[iC].tick_params(axis='both', labelsize=fontsize_tick)
        axs[iC].axhline(0, lw=0.5, color='grey', zorder=-10)  
        axs[iC].axvline(0, lw=0.5, color='grey', zorder=-10)
        axs[iC].spines['top'].set_visible(False)
        axs[iC].spines['right'].set_visible(False)
        axs[iC].set_ylim(-0.1, 1.1)
        axs[iC].axvspan(t[timepoints_onepulse[iC, 0]], t[timepoints_onepulse[iC, 1]], facecolor='lightgrey', edgecolor=None, alpha=0.5)
        if iC > 0:
            axs[iC].spines['left'].set_visible(False)
            axs[iC].set_yticks([])
        # axs[iC].set_xlabel('Time (s)', fontsize=fontsize_label)

    # save
    # axs[1, 0].legend(frameon=False, fontsize=fontsize_legend)
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig2A_data', dpi=600)
    plt.savefig(root + 'visualization/Fig2A_data.svg')
    plt.close()

def plot_dynamics(metric, dynamics_lin, dynamics_log, visual_area, bootstrapped, computation, tempCond, CI_low, CI_high, color, n_electrodes_per_area, t1_plot, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 15
    fontsize_tick           = 10

    # join dynamics
    dynamics = [dynamics_lin, dynamics_log]

    # initiate figure
    fig, axs = plt.subplots(1, len(visual_area), figsize=(10, 3))
    sns.despine(offset=10)

    linestyle = ['dashed', 'dashdot']
    color_fit = ['lightgrey', 'darkgrey']
    lw = 2

    offset              = [-5, 0, 5]
    for iVA, VA in enumerate(visual_area):

        # select data
        if bootstrapped:

            # select data
            data_current = metric[iVA, :, :]

            # compute spread
            data_mean = np.mean(data_current, 1)

        else:

            # select data
            data_current = metric[iVA]

            # compute spread
            data_mean = np.mean(data_current, 1)
            data_sem = np.std(data_current, 1)/math.sqrt(n_electrodes_per_area[iVA])
        
        # plot
        axs[iVA].scatter(tempCond+offset[iVA], data_mean, facecolor=color[iVA], edgecolor='white', s=80, zorder=1)

        # plot spread
        for iC in range(len(tempCond)):
            if bootstrapped:
                data_CI = np.nanpercentile(data_current[iC, :], [CI_low, CI_high])
                axs[iVA].plot([tempCond[iC]+offset[iVA], tempCond[iC]+offset[iVA]], [data_CI[0], data_CI[1]], color=color[iVA], zorder=-10)
            else:
                axs[iVA].plot([tempCond[iC]+offset[iVA], tempCond[iC]+offset[iVA]], [data_mean[iC] - data_sem[iC], data_mean[iC] + data_sem[iC]], color=color[iVA], zorder=-10)

        # plot dynamics fit
        for iComp, comp in enumerate(computation):

            # select data
            if bootstrapped:
                data_current = dynamics[iComp][iVA, :, :]
            else:
                data_current = dynamics[iComp][iVA][:, :]

            # compute
            data_mean = np.mean(data_current, 0)

            # plot
            axs[iVA].plot(t1_plot+offset[iVA], data_mean, color=color_fit[iComp], lw=lw, label=comp, linestyle=linestyle[iComp], zorder=-10)

        # adjust axes
        axs[iVA].tick_params(axis='both', labelsize=fontsize_tick)  
        axs[iVA].spines['top'].set_visible(False)
        axs[iVA].spines['right'].set_visible(False)
        axs[iVA].set_xticks(tempCond)
        axs[iVA].set_xticklabels(tempCond, rotation=45)
        # axs[iVA].set_title(VA, fontsize=fontsize_title)
        if iVA == 1:
            axs[iVA].set_xlabel('Duration (s)', fontsize=fontsize_label)
        if iVA == 0:    
            axs[iVA].set_ylabel('Response magnitude', fontsize=fontsize_label)

    # save fig
    axs[0].legend(fontsize=fontsize_legend, frameon=False)
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig2B_data', dpi=600)
    plt.savefig(root + 'visualization/Fig2B_data.svg')
    plt.close()

def plot_dynamics_metric(dynamics_fit, dynamics_ratio, visual_area, n_electrodes_per_area, bootstrapped, computation, CI_low, CI_high, color, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 15
    fontsize_tick           = 10

    # # initiate figure
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    sns.despine(offset=10)

    offset_comp     = [0, 1, 2]
    hatch           = ['/', None]
    alpha           = [0.7, 1]
    offset_VA       = [0, 4, 8]

    patchs = list()
    for iVA, VA in enumerate(visual_area):
        for iC, comp in enumerate(computation):

            # select data
            if bootstrapped:

                # select data
                data_current = dynamics_fit[iVA, :, iC]

                # compute
                data_mean = np.mean(data_current)
                data_CI = np.nanpercentile(data_current, [CI_low, CI_high])

            else:

                # select data
                data_current = dynamics_fit[iVA][:, iC]

                # compute
                data_mean = np.median(data_current)
                data_sem = np.std(data_current)/math.sqrt(n_electrodes_per_area[iVA])

            # plot
            patch = ax.bar(offset_VA[iVA]+offset_comp[iC], data_mean, color=color[iVA], edgecolor='white', label=VA, hatch=hatch[iC], alpha=alpha[iC])
            patchs.append(patch)

            # plot spread
            if bootstrapped:
                ax.plot([offset_VA[iVA]+offset_comp[iC], offset_VA[iVA]+offset_comp[iC]], [data_CI[0], data_CI[1]], color='black')
            else:
                ax.plot([offset_VA[iVA]+offset_comp[iC], offset_VA[iVA]+offset_comp[iC]], [data_mean - data_sem, data_mean], color='black')

    # adjust axes
    ax.tick_params(axis='both', labelsize=fontsize_tick)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([offset_VA[0]+0.5, offset_VA[1]+0.5, offset_VA[2]+0.5])
    ax.set_xticklabels(visual_area)
    ax.set_ylabel(r'R$^{2}$', fontsize=fontsize_label)

    # save figure
    # ax.legend([(patchs[2], patchs[3], patchs[4]), (patchs[1], patchs[3], patchs[5])], [computation[0], computation[1]], frameon=False, fontsize=fontsize_legend, 
    #           bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    # ax.legend(frameon=False, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig2C_data', dpi=600)
    plt.savefig(root + 'visualization/Fig2C_data.svg')
    plt.close()

def plot_regression(visual_area, color, ratio_trans_sust, dynamics_ratio, bootstrapped, CI_low, CI_high, n_electrodes_per_area, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 12
    fontsize_tick           = 10

    # initiate figure
    fig = plt.figure(figsize=(3, 3))
    gs = fig.add_gridspec(17, 17)
    ax = dict()

    # initiate axes
    ax['regression'] = fig.add_subplot(gs[3:15, :14])

    ax['avg_lin_log'] = fig.add_subplot(gs[:2, :14])
    ax['avg_trans_sust'] = fig.add_subplot(gs[3:15, 15:17])
    ax_avg = [ax['avg_lin_log'], ax['avg_trans_sust']]

    sns.despine(offset=10)

    # set ylim
    if bootstrapped:
        lmts = [[0.5, 1.4], [1, 6]]
    else:
        lmts = [[0.4, 1.7], [0, 9]]

    # statistics
    for i in range(2):
        if i == 0:
            print('Ratio lin/log')
            result = f_oneway(dynamics_ratio[0], dynamics_ratio[1], dynamics_ratio[2])
            print(result)
            res = tukey_hsd(dynamics_ratio[0], dynamics_ratio[1], dynamics_ratio[2])
            print(res)
        else:
            print('Ratio trans/sust')
            result = f_oneway(ratio_trans_sust[0], ratio_trans_sust[1], ratio_trans_sust[2])
            print(result)
            res = tukey_hsd(ratio_trans_sust[0], ratio_trans_sust[1], ratio_trans_sust[2])
            print(res)

    # visualize correlations
    for iVA in range(len(visual_area)):

        # select data
        if bootstrapped:
            data_current_x = dynamics_ratio[iVA, :]        
            data_current_y = ratio_trans_sust[iVA, :]
        else:
            data_current_x = dynamics_ratio[iVA]        
            data_current_y = ratio_trans_sust[iVA]

        # visualize
        if bootstrapped:
            ax['regression'].scatter(data_current_x, data_current_y, linewidths=0.75, facecolor=color[iVA], edgecolor='white', s=20)
        else:
            ax['regression'].scatter(data_current_x, data_current_y, linewidths=0.75, facecolor=color[iVA], edgecolor='white', s=30)#, alpha=0.7)

        # concat
        data_current = [data_current_x, data_current_y]

        for i in range(2): 

            # visualize averages
            if bootstrapped:
                
                # compute metrics
                data_mean = np.median(data_current[i])
                data_CI = np.nanpercentile(data_current[i], [CI_low, CI_high])

                if i == 0:
                    ax_avg[i].plot([data_CI[0], data_CI[1 ]], [iVA, iVA], color=color[iVA], zorder=-10)
                else:
                    ax_avg[i].plot([iVA, iVA], [data_CI[0], data_CI[1]], color=color[iVA], zorder=-10)

            else:

                # compute metrics
                data_mean = np.mean(data_current[i])
                data_sem = np.std(data_current[i])/math.sqrt(n_electrodes_per_area[iVA])

                if i == 0:
                    ax_avg[i].plot([data_mean - data_sem, data_mean + data_sem], [iVA, iVA], color=color[iVA], zorder=-10)
                else:
                    ax_avg[i].plot([iVA, iVA], [data_mean - data_sem, data_mean + data_sem], color=color[iVA], zorder=-10)

                # plot mean
                print(data_mean)
                if i == 0:
                    ax_avg[i].scatter(data_mean, iVA, facecolor='white', edgecolors=color[iVA], s=40, zorder=1)
                else:
                    ax_avg[i].scatter(iVA, data_mean, facecolor='white', edgecolors=color[iVA], s=40, zorder=1)

            # adjust axes
            ax_avg[i].set_xticks([])
            ax_avg[i].set_yticks([])
            if i == 0:
                ax_avg[i].spines['top'].set_visible(False)
                ax_avg[i].spines['right'].set_visible(False)
                ax_avg[i].spines['left'].set_visible(False)
                ax_avg[i].spines['bottom'].set_color('grey')
                ax_avg[i].set_xlim(lmts[i][0], lmts[i][1])
                ax_avg[i].set_ylim(-1, 3)
            else:
                ax_avg[i].spines['top'].set_visible(False)
                ax_avg[i].spines['bottom'].set_visible(False)
                ax_avg[i].spines['right'].set_visible(False)
                ax_avg[i].spines['left'].set_color('grey')
                ax_avg[i].set_ylim(lmts[i][0], lmts[i][1])
                ax_avg[i].set_xlim(-1, 3)

    # plot regression line
    if bootstrapped:
        x = dynamics_ratio[0, :].tolist() + dynamics_ratio[1, :].tolist() + dynamics_ratio[2, :].tolist()
        y = ratio_trans_sust[0, :].tolist() + ratio_trans_sust[1, :].tolist() + ratio_trans_sust[2, :].tolist()
    else:
        x = dynamics_ratio[0].tolist() + dynamics_ratio[1].tolist() + dynamics_ratio[2].tolist()
        y = ratio_trans_sust[0].tolist() + ratio_trans_sust[1].tolist() + ratio_trans_sust[2].tolist()

    # convert to array
    x = np.array(x)
    y = np.array(y)

    # fit linear regression
    model = LinearRegression().fit(x.reshape(-1, 1), y)

    # predict line
    y = model.intercept_ + model.coef_* np.linspace(np.min(x), np.max(x), 100)
    ax['regression'].plot(np.linspace(np.min(x), np.max(x), 100), y, color='grey', lw=2, linestyle='dashed')

    # adjust axes
    ax['regression'].tick_params(axis='both', labelsize=fontsize_tick)
    ax['regression'].spines['top'].set_visible(False)
    ax['regression'].spines['right'].set_visible(False)
    ax['regression'].set_ylabel('Ratio transient/sustained', fontsize=fontsize_label)
    ax['regression'].set_xlabel('Ratio fit lin/log', fontsize=fontsize_label)
    ax['regression'].set_xlim(lmts[0][0], lmts[0][1])
    ax['regression'].set_ylim(lmts[1][0], lmts[1][1])
    
    # savefigur
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig2D_data', dpi=600)
    plt.savefig(root + 'visualization/Fig2D_data.svg')
    plt.close()

def stats_lin_log(visual_area, bootstrapped, B_repetitions, dynamics_fit):

    alpha = 0.05

    for iVA, VA in enumerate(visual_area):

        # print progress
        print(VA)

        if bootstrapped:

            # ttest
            sample1 = dynamics_fit[iVA, :, 0]
            sample2 = dynamics_fit[iVA, :, 1]
            param_diffs = sample1 - sample2

            p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
            if p < len(visual_area):
                print('Linear vs. log fit', p, ' SIGNIFICANT')
            else:
                print('Linear vs. log fit', p)

        else:
            
            # ttest
            sample1 = dynamics_fit[iVA][:, 0]
            sample2 = dynamics_fit[iVA][:, 1]
            p = stats.ttest_rel(sample1, sample2)
            if p[1] < alpha/len(visual_area):
                print('Linear vs. log fit', p, ' SIGNIFICANT')
            else:
                print('Linear vs. log fit', p)

        print('\n')

def stats_regression(dynamics_ratio, bootstrapped, ratio_trans_sust):

    # oneway anova
    if bootstrapped:
        result = f_oneway(dynamics_ratio[0, :], dynamics_ratio[1, :], dynamics_ratio[2, :])
    else:
        result = f_oneway(dynamics_ratio[0][:], dynamics_ratio[1][:], dynamics_ratio[2][:])
    print(result)

    # convert to list
    if bootstrapped:
        x = dynamics_ratio[0, :].tolist() + dynamics_ratio[1, :].tolist() + dynamics_ratio[2, :].tolist()
        y = ratio_trans_sust[0, :].tolist() + ratio_trans_sust[1, :].tolist() + ratio_trans_sust[2, :].tolist()
    else:
        x = dynamics_ratio[0].tolist() + dynamics_ratio[1].tolist() + dynamics_ratio[2].tolist()
        y = ratio_trans_sust[0].tolist() + ratio_trans_sust[1].tolist() + ratio_trans_sust[2].tolist()

    # convert to array
    x = np.array(x)
    y = np.array(y)

    # fit linear regression
    model = LinearRegression().fit(x.reshape(-1, 1), y)

    # slope and statistics
    slope = model.coef_[0]
    print("Slope:", slope)

    x_with_const = sm.add_constant(x)
    model_sm = sm.OLS(y, x_with_const).fit()
    print(model_sm.summary())





