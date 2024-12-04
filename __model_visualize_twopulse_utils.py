import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

def plot_broadband_all(data, trials_lbls, start, tempCond, duration, n_layer, n_img, population, dataset, output_mode, color_tempCond, color_layer, color_trial, root_data, root_vis):

    # fontsizes 
    fontsize_title          = 8
    fontsize_legend         = 6
    fontsize_label          = 6
    fontsize_tick           = 5

    # network activations
    start_plot = 2

    # initiate figure
    fig, axs = plt.subplots(n_layer, len(tempCond), figsize=(10, 5))
    fig.tight_layout(h_pad=2)
    sns.despine(offset=10)

    # PLOT AVERAGE
    lbls = []
    for iT, trial in enumerate(trials_lbls):
        for iL in range(n_layer):

            # set label
            if population != None:
                lbl = output_mode + str(iL+1) + population
            else:
                lbl = output_mode + str(iL+1)
            lbls.append(lbl)

            for iC in range(len(tempCond)):

                # select data
                data_current = data[iT, iL, iC, :, start_plot:]#/np.max(data[iT, iL, -1, :, start_plot:]).reshape(-1, 1)

                # normalize data
                # for iImg in range(n_img):
                    # data_current_min = np.min(data_current[iImg, :])
                    # data_current_max = np.max(data_current[iImg, :])
                    # data_current[iImg, :] = (data_current[iImg, :] - data_current_min) / (data_current_max - data_current_min)

                    # data_avg = np.min(abs(data_current[iImg, start-3]))
                    # data_current[iImg, :] = abs(data_current[iImg, :]) - data_avg

                    # axs[iL, iC].plot(data_current[iImg, :], color=color[iT], lw=0.2)
                
                # compute
                data_mean = np.mean(data_current, 0)

                # visualize
                axs[iL, iC].plot(data_mean, color=color_trial[iT], markersize=1, marker='o', lw=0.5)

                # plot stimulus
                axs[iL, iC].tick_params(axis='both', labelsize=fontsize_tick)
                axs[iL, iC].axhline(0, lw=0.5, color='grey')   
                axs[iL, iC].spines['top'].set_visible(False)
                axs[iL, iC].spines['right'].set_visible(False)
                # axs[iL, iC].set_ylim(-0.1, 1.1)
                axs[iL, iC].axvspan(start-start_plot, start-start_plot+duration, facecolor='lightgrey', edgecolor=None, alpha=0.5)
                axs[iL, iC].axvspan(start-start_plot+tempCond[iC]+duration, start-start_plot+tempCond[iC]+2*duration, facecolor='lightgrey', edgecolor=None, alpha=0.5)

                if iL == 0:
                    axs[iL, iC].set_title(str(int(tempCond[iC])) + ' model timestp', fontsize=fontsize_title)
                if iL == n_layer - 1:
                    axs[iL, iC].set_xlabel('Time (s)', fontsize=fontsize_label)
                else:
                    axs[iL, iC].spines['bottom'].set_visible(False)
                    axs[iL, iC].set_xticks([])
                if (iL == n_layer - 1) & (iC > 0):
                    axs[iL, iC].spines['left'].set_visible(False)
                if iC > 0:
                    axs[iL, iC].spines['right'].set_visible(False)
                    axs[iL, iC].set_yticks([])
                if iC == 0:
                    axs[iL, iC].set_ylabel(lbls[iL], color=color_layer[iL], weight='bold', fontsize=fontsize_title)
        
    # save
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(root_vis + 'twopulse', dpi=600)
    plt.close()

def plot_broadband(data, data_one_pulse, data_two_pulse, trials_lbls, start, tempCond, duration, n_img, color_trial, root_vis):

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
    iL = 1

    # PLOT AVERAGE
    for iT, _ in enumerate(trials_lbls):
        for iC in range(len(tempCond)):

            # select data
            data_current = data[iT, iL, iC, :, start_plot:] #- data[iT, iL, iC, :, -1].reshape(-1, 1)

            # # normalize data
            for iImg in range(n_img):

                # normalize
                data_current[iImg, :] = data_current[iImg, :]#/np.max(data_current[iImg, :])

                axs[iC].plot(np.arange(data_current.shape[1])-start_plot, data_current[iImg, :], color=color_trial[iT], lw=0.25, alpha=0.3)
            
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
    plt.savefig(root_vis + 'twopulse_8_duration', dpi=600)
    plt.savefig(root_vis + 'twopulse_8_duration.svg')
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

            # plot spread
            for iC in range(len(tempCond)):

                # if iT == 0:
                #     print('Contrast: ', iC)
                #     sample1 = metric[0, iL, iC, :]
                #     print(np.mean(sample1))
                #     sample2 = metric[1, iL, iC, :]
                #     print(np.mean(sample2))
                #     p = mannwhitneyu(sample1, sample2)[1]
                #     if p < 0.05:
                #         print('repetition vs. alternation', p, ' SIGNIFICANT')
                #     else:
                #         print('repetition vs. alternation', p)

                # select data
                data_current = metric[iT, iL, iC, :]

                # compute spread
                data_mean = np.mean(data_current)
                data_sem = np.std(data_current)/math.sqrt(n_img)
                
                # plot
                axs[iL].scatter(tempCond[iC]+offset_trial[iT], data_mean, facecolor=color_trial[iT], edgecolor='white', s=50, zorder=1)

                # plot spread
                axs[iL].plot([tempCond[iC]+offset_trial[iT], tempCond[iC]+offset_trial[iT]], [data_mean - data_sem, data_mean + data_sem], color=color_trial[iT], zorder=-10)

                # plot spread
                sns.stripplot(x=np.ones(n_img)*(tempCond[iC]+offset_trial[iT]), y=metric[iT, iL, iC, :], jitter=True, ax=axs[iL], color=color_trial[iT], size=2, alpha=0.2, native_scale=True)

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
        axs[iL].set_ylim(0.4, 2)
        # if iL == 0:    
        #     axs[iL].set_ylabel('Response magnitude', fontsize=fontsize_label)
        # else:
        #     axs[iL].set_yticklabels([])

    # save fig
    plt.tight_layout()
    plt.savefig(root + 'twopulse_dynamics_fit_8_duration', dpi=600)
    plt.savefig(root + 'twopulse_dynamics_fit_8_duration.svg')
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
    plt.savefig(root + 'twopulse_dynamics_fit_metric_8_duration', dpi=600)
    plt.savefig(root + 'twopulse_dynamics_fit_metric_8_duration.svg')
    plt.close()

def plot_regression_CEandSC(ratio_lin_log, metric, CEandSC_values, n_layer, color, n_img, model, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 20
    fontsize_tick           = 15

    # initiate figure
    fig, axs = plt.subplots(2, 4, figsize=(17, 7))
    
    sns.despine(offset=10)

    # set title
    # fig.suptitle(model, fontsize=fontsize_title)

    # summary statistics
    summary_statistics  = ['CE', 'SC']
    color               = ['dodgerblue', 'forestgreen']

    # visualize correlations
    for iL in range(n_layer):
    # for iL in range(1):

        # row numbers
        if (iL == 0) | (iL == 1):
            row = 0
        elif (iL == 2) | (iL == 3):
            row = 1

        iL_col = iL%2

        for iSumStat, SumStat in enumerate(summary_statistics):

            # select data
            data_current_x = CEandSC_values[:, iSumStat]
            data_current_y = np.mean(metric[0, iL, :2, :], 0)

            # visualize
            axs[row, 2*iL_col+iSumStat].scatter(data_current_x, data_current_y, color='grey', facecolor='white', s=100, linewidth=1.5)#, alpha=0.7)

            # select data
            x = data_current_x
            y = data_current_y

            # fit linear regression
            model = LinearRegression().fit(x.reshape(-1, 1), y)

            # slope and statistics
            slope = model.coef_[0]
            print("Slope:", slope)

            x_with_const = sm.add_constant(x)
            model_sm = sm.OLS(y, x_with_const).fit()
            print('Layer: ', iL+1, ', statistic: ', SumStat)
            print(model_sm.summary())
            # print(model_sm._results)

            # predict line
            y = model.intercept_ + model.coef_*np.linspace(np.min(x), np.max(x), 100)
            axs[row, 2*iL_col+iSumStat].plot(np.linspace(np.min(x), np.max(x), 100), y, color='lightsalmon', lw=4, linestyle='dashed')

            # adjust axes
            axs[row, 2*iL_col+iSumStat].tick_params(axis='both', labelsize=fontsize_tick)
            axs[row, 2*iL_col+iSumStat].spines['top'].set_visible(False)
            axs[row, 2*iL_col+iSumStat].spines['right'].set_visible(False)
            axs[row, 2*iL_col+iSumStat].set_xticks([])
            if row == 1:
                axs[row, 2*iL_col+iSumStat].set_xlabel(SumStat, fontsize=fontsize_label)
            if (iSumStat == 0) & ((iL == 0) | (iL == 2)):
                axs[row, 2*iL_col+iSumStat].set_ylabel('Recovery', fontsize=fontsize_label)
            if iSumStat == 1:
                axs[row, 2*iL_col+iSumStat].set_yticks([])
    
    # save figure
    plt.tight_layout()
    plt.savefig(root + 'twopulse_dynamics_regression_CEandSC.png', dpi=600)
    plt.savefig(root + 'twopulse_dynamics_regression_CEandSC.svg')
    plt.close()

def stats_trial(metric, n_layer):

    alpha = 0.05

    for iL in range(n_layer):

        # print progress
        print('Layer', iL+1)

        # ttest (non parametric)
        # sample1 = adaptation_avg[0, iL, :]
        # sample2 = adaptation_avg[1, iL, :]
        # p = stats.ttest_rel(sample1, sample2)[1]
        sample1 = metric[0, iL, -1, :]
        sample2 = metric[1, iL, -1, :]
        p = mannwhitneyu(sample1, sample2)[1]
        if p < alpha:
            print('repetition vs. alternation', p, ' SIGNIFICANT')
        else:
            print('repetition vs. alternation', p)

        print('\n')