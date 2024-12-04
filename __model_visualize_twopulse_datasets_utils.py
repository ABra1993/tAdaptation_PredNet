import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
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

def plot_broadband(data, dataset, trials_lbls, start, tempCond, duration, n_img, color_trial, root_vis):

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
    plt.savefig(root_vis + 'twopulse_' + dataset, dpi=600)
    plt.savefig(root_vis + 'twopulse_' + dataset + '.svg')
    plt.close()

def plot_broadband_all_fps(training_sets, dataset, data, data_one_pulse, data_two_pulse, trials_lbls, start, tempCond, duration, n_img, color_trial, root_vis):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 15
    fontsize_tick           = 10

    # initiate figure
    fig, axs = plt.subplots(len(training_sets), len(tempCond), figsize=(17, 2*len(training_sets)))
    sns.despine(offset=10)

    # start
    start_plot = 2

    # select layer
    iL = 0

    # PLOT AVERAGE
    for iTS, training_set in enumerate(training_sets):
        for iT, _ in enumerate(trials_lbls):
            for iC in range(len(tempCond)):

                # select data
                data_current = data[iTS, iT, iL, iC, :, start_plot:] #- data[iTS, iT, iL, iC, :, -1].reshape(-1, 1)

                # normalize data
                for iImg in range(n_img):

                    axs[iTS, iC].plot(np.arange(data_current.shape[1])-start_plot, data_current[iImg, :], color=color_trial[iT], lw=0.2, alpha=0.2)
                
                # compute
                data_mean = np.mean(data_current, 0)

                # visualize
                axs[iTS, iC].plot(np.arange(data_current.shape[1])-start_plot, data_mean, color=color_trial[iT], markersize=1, marker='o', lw=1)

                # adjust axes
                axs[iTS, iC].tick_params(axis='both', labelsize=fontsize_tick)
                axs[iTS, iC].axhline(0, lw=0.5, color='grey')
                axs[iTS, iC].axvline(0, lw=0.5, color='grey')   
                axs[iTS, iC].spines['top'].set_visible(False)
                axs[iTS, iC].spines['right'].set_visible(False)
                axs[iTS, iC].set_xticks([])
                if (iC > 0):
                    axs[iTS, iC].spines['left'].set_visible(False)
                    axs[iTS, iC].spines['right'].set_visible(False)
                    axs[iTS, iC].set_yticks([])
                if iT == 0:
                    axs[iTS, iC].axvspan(0, duration, facecolor='lightgrey', edgecolor=None, alpha=0.5)
                    axs[iTS, iC].axvspan(tempCond[iC]+duration, tempCond[iC]+2*duration, facecolor='lightgrey', edgecolor=None, alpha=0.5)

    # save
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(root_vis + 'twopulse_fps_' + dataset, dpi=600)
    plt.savefig(root_vis + 'twopulse_fps_' + dataset + '.svg')
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
    plt.savefig(root + 'twopulse_dynamics_fit', dpi=600)
    plt.savefig(root + 'twopulse_dynamics_fit.svg')
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
    plt.savefig(root + 'twopulse_dynamics_fit_metric', dpi=600)
    plt.savefig(root + 'twopulse_dynamics_fit_metric.svg')
    plt.close()

def plot_dynamics_all_dataset(training_sets, analyse, dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 17
    fontsize_tick           = 12

    # initiate figure
    _, axs = plt.subplots(len(training_sets), n_layer, figsize=(12, 3*len(training_sets)), sharey=True)
    sns.despine(offset=10)

    linestyle = ['solid', 'dashdot']
    color_fit = ['lightgrey', 'darkgrey']
    lw = 0.75

    offset_trial    = [-0.5, 0.5]
    for iTS, training_set in enumerate(training_sets):
        for iL in range(n_layer):
            for iT, trial in enumerate(trials):

                # select data
                data_current = metric[iTS, iT, iL, :, :]

                # compute spread
                data_mean = np.mean(data_current, 1)
                data_sem = np.std(data_current, 1)/math.sqrt(n_img)
                
                # plot
                axs[iTS, iL].scatter(tempCond+offset_trial[iT], data_mean, facecolor=color_trial[iT], edgecolor='white', s=80, zorder=1)

                # # plot spread
                # for iC in range(len(tempCond)):
                #     axs[iTS, iL].plot([tempCond[iC]+offset_trial[iT], tempCond[iC]+offset_trial[iT]], [data_mean[iC] - data_sem[iC], data_mean[iC] + data_sem[iC]], color=color_trial[iT], zorder=-10)

                # select data
                data_current = dynamics_log[iTS, iT, iL, :, :]

                # compute
                data_mean = np.mean(data_current, 0)

                # plot
                axs[iTS, iL].plot(t1_plot, data_mean, color=color_trial[iT], lw=lw, linestyle=linestyle[1], zorder=-10)

            # adjust axes
            axs[iTS, iL].tick_params(axis='both', labelsize=fontsize_tick)  
            axs[iTS, iL].spines['top'].set_visible(False)
            axs[iTS, iL].spines['right'].set_visible(False)
            axs[iTS, iL].set_xticks(tempCond)
            axs[iTS, iL].set_xticklabels(tempCond, rotation=45)
            axs[iTS, iL].axhline(1, lw=2, linestyle='--', zorder=-10, color='darkgrey')
            # axs[iTS, iL].set_ylim(0.4, 1.2)
            # if iL == 0:    
            #     axs[iL].set_ylabel('Response magnitude', fontsize=fontsize_label)
            # else:
            #     axs[iL].set_yticklabels([])

    # save fig
    plt.tight_layout()
    plt.savefig(root + 'twopulse_dynamics_fit', dpi=600)
    plt.savefig(root + 'twopulse_dynamics_fit.svg')
    plt.close()

def plot_dynamics_metric_all_dataset(training_sets, analyse, dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 17
    fontsize_tick           = 12

    # initiate figure

    fig, axs = plt.subplots(1, len(training_sets), figsize=(8, 3), sharey=True)
    sns.despine(offset=10)

    linestyle = ['solid', 'dashdot']
    color_fit = ['lightgrey', 'darkgrey']
    lw = 2

    # neural data
    neural_data_mean_same = 0.5859472802324072
    neural_data_mean_different = 0.7347749538702524
    neural_data_mean = [neural_data_mean_same, neural_data_mean_different]
    neural_data_color = ['royalblue', 'darkgoldenrod']

    # average activations
    avg_act = []

    # visualize
    offset_L        = [0, 2, 4, 6]
    offset_trial    = [-0.2, 0.2]
    for iTS, training_set in enumerate(training_sets):

        # plot average
        data_current = adaptation_avg[iTS, :, :, :].mean(0).mean(0)

        # compute
        data_mean = np.median(data_current)
        data_sem = np.std(data_current)/math.sqrt(n_img)
        avg_act.append(data_current)

        # axs[iTS].axhline(data_mean, color='salmon', zorder=-20)
        # axs[iTS].axhspan(data_mean - data_sem, data_mean + data_sem, edgecolor='white', facecolor='salmon', alpha=0.1, zorder=-20)

        print(training_set, data_mean)
        print('---------', training_set, data_sem)

        for iL in range(n_layer):
            for iT, trial in enumerate(trials):

                #### PLOT METRIC
                # ax.set_xticks([offset_L[0], offset_L[1], offset_L[2]])

                # select data
                data_current = adaptation_avg[iTS, iT, iL, :]

                # compute
                data_mean = np.median(data_current)
                data_sem = np.std(data_current)/math.sqrt(n_img)

                # plot data mean
                axs[iTS].scatter(offset_L[iL]+offset_trial[iT], neural_data_mean[iT], color=neural_data_color[iT], linewidths=1, marker='_', zorder=-3)

                # plot
                axs[iTS].scatter(offset_L[iL]+offset_trial[iT], data_mean, color=color_trial[iT], edgecolor='white', s=40, zorder=0)

                # plot spread (sem)
                axs[iTS].plot([offset_L[iL]+offset_trial[iT], +offset_L[iL]+offset_trial[iT]], [data_mean - data_sem, data_mean], color=color_trial[iT], zorder=-10)

                # plot individual electrodes
                sns.stripplot(x=np.ones(n_img)*(offset_L[iL]+offset_trial[iT]), y=adaptation_avg[iTS, iT, iL, :], jitter=True, ax=axs[iTS], color=color_trial[iT], size=2, alpha=0.35, native_scale=True)

        # adjust axes
        axs[iTS].tick_params(axis='both', labelsize=fontsize_tick)  
        axs[iTS].spines['top'].set_visible(False)
        axs[iTS].spines['right'].set_visible(False)
        axs[iTS].set_xticks(offset_L)
        axs[iTS].set_xticklabels(['', '', '', ''])
        axs[iTS].axhline(1, lw=1.5, linestyle='--', zorder=-10, color='silver') 
        axs[iTS].set_ylim(0.4, 2)
        # if iTS == 0:
        #     axs[iTS].set_ylabel('Avg. adaptation', fontsize=fontsize_label)

    # apply statistics
    result = f_oneway(avg_act[0], avg_act[1], avg_act[2], avg_act[3])
    print(result)
    res = tukey_hsd(avg_act[0], avg_act[1], avg_act[2], avg_act[3])
    print(res)
    
    # save fig
    plt.tight_layout()
    plt.savefig(root + 'twopulse_dynamics_fit_metric_' + analyse, dpi=600)
    plt.savefig(root + 'twopulse_dynamics_fit_metric_' + analyse + '.svg')
    plt.close()

def plot_dynamics_metric_all_datasets_fps_loss(training_sets, analyse, datasets, dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 15
    fontsize_label          = 18
    fontsize_tick           = 16

    # initiate figure

    fig, axs = plt.subplots(len(datasets), len(training_sets), figsize=(11, 10), sharey=True)
    sns.despine(offset=10)

    linestyle = ['solid', 'dashdot']
    color_fit = ['lightgrey', 'darkgrey']
    lw = 2

    # average activation
    avg_act = []

    # neural data
    neural_data_mean_same = 0.5859472802324072
    neural_data_mean_different = 0.7347749538702524
    neural_data_mean = [neural_data_mean_same, neural_data_mean_different]
    neural_data_color = ['royalblue', 'darkgoldenrod']

    # visualize
    offset_L        = [0, 1, 2, 3]
    offset_trial    = [-0.2, 0.2]
    for iD, dataset in enumerate(datasets):
        for iTS, training_set in enumerate(training_sets):

            # plot average
            data_current = adaptation_avg[iD, iTS, :, :, :].mean(0).mean(0)
            avg_act.append(data_current)

            # compute
            data_mean = np.median(data_current)
            data_sem = np.std(data_current)/math.sqrt(n_img)

            # axs[iD, iTS].axhline(data_mean, color='salmon', zorder=-20)
            # axs[iD, iTS].axhspan(data_mean - data_sem, data_mean + data_sem, edgecolor='white', facecolor='salmon', alpha=0.1, zorder=-20)

            # print(training_set, data_mean)
            # print('---------', training_set, data_sem)

            for iL in range(n_layer):
                for iT, trial in enumerate(trials):

                    #### PLOT METRIC
                    # ax.set_xticks([offset_L[0], offset_L[1], offset_L[2]])

                    # select data
                    data_current = adaptation_avg[iD, iTS, iT, iL, :]

                    # compute
                    data_mean = np.median(data_current)
                    data_sem = np.std(data_current)/math.sqrt(n_img)

                    # plot data mean
                    axs[iD, iTS].scatter(offset_L[iL]+offset_trial[iT], neural_data_mean[iT], color=neural_data_color[iT], s=40, marker='_', zorder=-3)

                    # plot
                    axs[iD, iTS].scatter(offset_L[iL]+offset_trial[iT], data_mean, color=color_trial[iT], edgecolor='white', s=100)

                    # plot spread (sem)
                    axs[iD, iTS].plot([offset_L[iL]+offset_trial[iT], +offset_L[iL]+offset_trial[iT]], [data_mean - data_sem, data_mean], color=color_trial[iT], zorder=-10)

                    # plot individual electrodes
                    sns.stripplot(x=np.ones(n_img)*(offset_L[iL]+offset_trial[iT]), y=adaptation_avg[iD, iTS, iT, iL, :], jitter=True, ax=axs[iD, iTS], color=color_trial[iT], size=4, alpha=0.2, native_scale=True)

                # statistics
                print(dataset,', training set ', iTS + 1, ', layer ', iL+1)
                results = stats.ttest_rel(adaptation_avg[iD, iTS, 0, iL, :], adaptation_avg[iD, iTS, 1, iL, :])
                if results[1] < 0.05:
                    print(results)

            # adjust axes
            axs[iD, iTS].tick_params(axis='both', labelsize=fontsize_tick)  
            axs[iD, iTS].spines['top'].set_visible(False)
            axs[iD, iTS].spines['right'].set_visible(False)
            axs[iD, iTS].set_xticks(offset_L)
            axs[iD, iTS].set_xticklabels(['', '', '', ''])
            axs[iD, iTS].axhline(1, lw=2, linestyle='--', zorder=-10, color='darkgrey')
            axs[iD, iTS].set_ylim(0.4, 2.2)

    # save fig
    plt.tight_layout()
    plt.savefig(root + 'twopulse_dynamics_fit_metric_' + analyse, dpi=600)
    plt.savefig(root + 'twopulse_dynamics_fit_metric_' + analyse + '.svg')
    plt.close()

def plot_dynamics_metric_all_fps_loss(training_sets, analyse, dataset, dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 15
    fontsize_label          = 18
    fontsize_tick           = 16

    # initiate figure

    fig, axs = plt.subplots(1, len(training_sets), figsize=(8, 2.5), sharey=True)
    sns.despine(offset=10)

    linestyle = ['solid', 'dashdot']
    color_fit = ['lightgrey', 'darkgrey']
    lw = 2

    # average activation
    avg_act = []

    # visualize
    offset_L        = [0, 1, 2, 3]
    offset_trial    = [-0.2, 0.2]
    for iTS, training_set in enumerate(training_sets):

        # plot average
        data_current = adaptation_avg[iTS, :, :, :].mean(0).mean(0)
        avg_act.append(data_current)

        # compute
        data_mean = np.median(data_current)
        data_sem = np.std(data_current)/math.sqrt(n_img)

        axs[iTS].axhline(data_mean, color='salmon', zorder=-20)
        axs[iTS].axhspan(data_mean - data_sem, data_mean + data_sem, edgecolor='white', facecolor='salmon', alpha=0.1, zorder=-20)

        print(training_set, data_mean)
        print('---------', training_set, data_sem)

        for iL in range(n_layer):
            for iT, trial in enumerate(trials):

                #### PLOT METRIC
                # ax.set_xticks([offset_L[0], offset_L[1], offset_L[2]])

                # select data
                data_current = adaptation_avg[iTS, iT, iL, :]

                # compute
                data_mean = np.median(data_current)
                data_sem = np.std(data_current)/math.sqrt(n_img)

                # plot
                axs[iTS].scatter(offset_L[iL]+offset_trial[iT], data_mean, color=color_trial[iT], edgecolor='white', s=100)

                # plot spread (sem)
                axs[iTS].plot([offset_L[iL]+offset_trial[iT], +offset_L[iL]+offset_trial[iT]], [data_mean - data_sem, data_mean], color=color_trial[iT], zorder=-10)

                # plot individual electrodes
                sns.stripplot(x=np.ones(n_img)*(offset_L[iL]+offset_trial[iT]), y=adaptation_avg[iTS, iT, iL, :], jitter=True, ax=axs[iTS], color=color_trial[iT], size=4, alpha=0.2, native_scale=True)

        # adjust axes
        axs[iTS].tick_params(axis='both', labelsize=fontsize_tick)  
        axs[iTS].spines['top'].set_visible(False)
        axs[iTS].spines['right'].set_visible(False)
        axs[iTS].set_xticks(offset_L)
        axs[iTS].set_xticklabels(['', '', '', ''])
        axs[iTS].axhline(1, lw=2, linestyle='--', zorder=-10, color='darkgrey')
        axs[iTS].set_ylim(0.4, 2.2)
        # if iTS == 0:
        #     axs[iTS].set_ylabel('Avg. adaptation', fontsize=fontsize_label)

    # statistics
    if len(training_sets) > 2:
        result = f_oneway(avg_act[0], avg_act[1], avg_act[2])
        print(result)
        res = tukey_hsd(avg_act[0], avg_act[1], avg_act[2])
        print(res)
    else:
        results = stats.ttest_rel(avg_act[0], avg_act[1])
        print(results)

    # save fig
    plt.tight_layout()
    plt.savefig(root + 'twopulse_dynamics_fit_metric_' + analyse + '_' + dataset, dpi=600)
    plt.savefig(root + 'twopulse_dynamics_fit_metric_' + analyse + '_' + dataset + '.svg')
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

def stats_trial(training_sets, adaptation_avg, n_layer):

    alpha = 0.05
    for iTS, training_set in enumerate(training_sets):

        print(30*'-')
        print(training_set)
        print(30*'-')
        
        for iL in range(n_layer):

            # print progress
            print('Layer', iL+1)

            # ttest
            sample1 = adaptation_avg[iTS, 0, iL, :]
            sample2 = adaptation_avg[iTS, 1, iL, :]
            # p = stats.ttest_rel(sample1, sample2)[1]
            p = mannwhitneyu(sample1, sample2, method="exact")[1]
            if p < alpha:
                print('repetition vs. alternation', p, ' SIGNIFICANT')
            else:
                print('repetition vs. alternation', p)

            print('\n')