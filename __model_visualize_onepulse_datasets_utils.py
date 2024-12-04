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

def plot_dynamics_metric(train_set, dynamics_fit, n_layer, computation, color, n_img, init, output_mode, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 15
    fontsize_tick           = 10

    # # initiate figure
    fig = plt.figure(figsize=(8, 3))
    ax = plt.gca()
    sns.despine(offset=10)

    barWitdh        = 0.5
    hatch           = ['//', None]
    alpha           = [0.7, 1]
    offset_L        = [0, 15, 30, 45]
    offset_comp     = [0, 5, ]

    patchs = list()
    for iL in range(n_layer):
        for iInit in range(init):

            # statistical analysis
            sample1 = dynamics_fit[:, iL, :, 0].mean(0)
            sample2 = dynamics_fit[:, iL, :, 1].mean(0)
            result = stats.ttest_rel(sample1, sample2)
            print(result)

            for iC, comp in enumerate(computation):

                # select data
                data_current = dynamics_fit[iInit, iL, :, iC]

                # compute
                data_mean = np.mean(data_current)
                data_sem = np.std(data_current)/math.sqrt(init)

                # plot
                patch = ax.bar(offset_L[iL]+offset_comp[iC]+iInit, data_mean, color=color[iL], edgecolor='white', label=output_mode + str(iL+1), hatch=hatch[iC], alpha=0.3+0.15*iInit)
                patchs.append(patch)

                # plot spread
                ax.plot([offset_L[iL]+offset_comp[iC]+iInit, offset_L[iL]+offset_comp[iC]+iInit], [data_mean - data_sem, data_mean], color='black')

    # adjust axes
    ax.tick_params(axis='both', labelsize=fontsize_tick)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([offset_L[0]+0.5+init-1, offset_L[1]+0.5+init-1, offset_L[2]+0.5+init-1, offset_L[3]+0.5+init-1])
    ax.set_xticklabels(['E1', 'E2', 'E3', 'E4'])
    ax.set_ylabel(r'R$^{2}$', fontsize=fontsize_label)

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'perInit_onepulse_dynamics_fit_metric_static', dpi=600)
    # plt.savefig(root + 'perInit_onepulse_dynamics_fit_metric.svg')

def plot_regression(train_sets, ratio_lin_log, ratio_trans_sust, n_layer, color, n_img, init, root):

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

    # # statistics
    # print('\n')
    # for i in range(2):
    #     if i == 0:
    #         print('Ratio lin/log')
    #         result = f_oneway(ratio_lin_log[:, :, 0, :].mean(0).mean(0), ratio_lin_log[:, :, 1, :].mean(0).mean(0), ratio_lin_log[:, :, 2, :].mean(0).mean(0), ratio_lin_log[:, :, 3, :].mean(0).mean(0))
    #         print(result)
    #         res = tukey_hsd(ratio_lin_log[:, :, 0, :].mean(0).mean(0), ratio_lin_log[:, :, 1, :].mean(0).mean(0), ratio_lin_log[:, :, 2, :].mean(0).mean(0), ratio_lin_log[:, :, 3, :].mean(0).mean(0))
    #         print(res)
    #     else:
    #         print('Ratio trans/sust')
    #         result = f_oneway(ratio_trans_sust[:, :, 0, :].mean(0).mean(0), ratio_trans_sust[:, :, 1, :].mean(0).mean(0), ratio_trans_sust[:, :, 2, :].mean(0).mean(0), ratio_trans_sust[:, :, 3, :].mean(0).mean(0))
    #         print(result)
    #         res = tukey_hsd(ratio_trans_sust[:, :, 0, :].mean(0).mean(0), ratio_trans_sust[:, :, 1, :].mean(0).mean(0), ratio_trans_sust[:, :, 2, :].mean(0).mean(0), ratio_trans_sust[:, :, 3, :].mean(0).mean(0))
    #         print(res)

    # set x and y limits
    # lmts = [[0.5, 1.3], [-5, 60]]

    # visualize correlations
    for iT, train_set in enumerate(train_sets):

        # initiate dataframe
        data_per_layer = []

        for iL in range(n_layer):

            # select data
            data_current_x = np.mean(ratio_lin_log[iT, :, iL, :], 0)
            data_current_y = np.mean(ratio_trans_sust[iT, :, iL, :], 0)

            # visualize
            print(color[iT])
            ax['regression'].scatter(data_current_x, data_current_y, facecolor=color[iT], linewidths=0.5, edgecolor='white', s=20) #, alpha=0.5)

            # concat
            data_current = [data_current_x, data_current_y]
            data_per_layer.append(data_current)

        # # visualize averages
        for iL in range(n_layer):
            for i in range(2): 

        #         # visualize averages
        #         data_mean = np.mean(data_per_layer[iL][i])
        #         data_sem = np.std(data_per_layer[iL][i])/math.sqrt(init)

        #         if i == 0:
        #             ax_avg[i].plot([data_mean - data_sem, data_mean + data_sem], [iL, iL], color=color[iT], lw=0.75, zorder=-10)
        #         else:
        #             ax_avg[i].plot([iL, iL], [data_mean - data_sem, data_mean + data_sem], color=color[iT], lw=0.75, zorder=-10)

        #         # plot mean
        #         if i == 0:
        #             ax_avg[i].scatter(data_mean, iL, edgecolors=color[iT], facecolor='white', s=20, zorder=1)
        #         else:
        #             ax_avg[i].scatter(iL, data_mean, edgecolors=color[iT], facecolor='white', linewidth=1, s=20, zorder=1)

                # adjust axes
                ax_avg[i].set_xticks([])
                ax_avg[i].set_yticks([])      
                if i == 0:
                    ax_avg[i].spines['top'].set_visible(False)
                    ax_avg[i].spines['right'].set_visible(False)
                    ax_avg[i].spines['left'].set_visible(False)
                    ax_avg[i].spines['bottom'].set_color('grey')
                    # ax_avg[i].set_xlim(lmts[i][0], lmts[i][1])
                    ax_avg[i].set_ylim(-1, 4)
                else:
                    ax_avg[i].spines['top'].set_visible(False)
                    ax_avg[i].spines['bottom'].set_visible(False)
                    ax_avg[i].spines['right'].set_visible(False)
                    ax_avg[i].spines['left'].set_color('grey')
        #             ax_avg[i].set_ylim(lmts[i][0], lmts[i][1])
                    ax_avg[i].set_xlim(-1, 4)

        # ax['regression'].set_xlim(lmts[0][0], lmts[0][1])
        # ax['regression'].set_ylim(lmts[1][0], lmts[1][1])

        ################################################################### ALL

        # plot regression line
        x = ratio_lin_log[iT, :, 0, :].mean(0).tolist() + ratio_lin_log[iT, :, 1, :].mean(0).tolist() + ratio_lin_log[iT, :, 2, :].mean(0).tolist() + ratio_lin_log[iT, :, 3, :].mean(0).tolist()
        y = ratio_trans_sust[iT, :, 0, :].mean(0).tolist() + ratio_trans_sust[iT, :, 1, :].mean(0).tolist() + ratio_trans_sust[iT, :, 2, :].mean(0).tolist() + ratio_trans_sust[iT, :, 3, :].mean(0).tolist()

        # convert to array
        x = np.array(x)
        y = np.array(y)

        # fit linear regression
        model_LR = LinearRegression().fit(x.reshape(-1, 1), y)

        # slope and statistics
        slope = model_LR.coef_[0]
        print("Slope:", slope)

        x_with_const = sm.add_constant(x)
        model_sm = sm.OLS(y, x_with_const).fit()
        print(model_sm.summary())

        # predict line
        y = model_LR.intercept_ + model_LR.coef_* np.linspace(np.min(x), np.max(x), 100)
        ax['regression'].plot(np.linspace(np.min(x), np.max(x), 100), y, color=color[iT], lw=2, linestyle='dashed')

        # adjust axes
        ax['regression'].tick_params(axis='both', labelsize=fontsize_tick)
        ax['regression'].spines['top'].set_visible(False)
        ax['regression'].spines['right'].set_visible(False)
        ax['regression'].set_ylabel('Ratio transient/sustained', fontsize=fontsize_label)
        ax['regression'].set_xlabel('Ratio fit lin/log', fontsize=fontsize_label)

    # savefigur
    plt.savefig(root + 'onepulse_dynamics_regression', dpi=600)
    plt.savefig(root + 'onepulse_dynamics_regression.svg')
    plt.close()

def plot_regression_dataset_staticmotion(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, input_motion, n_layer, color, n_img, init, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 12
    fontsize_tick           = 10

    # initiate figure
    fig, axs = plt.subplots(len(input_motion), len(training_sets)-1, figsize=(6, 3))

    # initiate axes
    sns.despine(offset=10)

    # markers
    marker = ['o', 'v', '^', '*', 'X']

    # plot untrained network
    # for iTS, training_set in enumerate(training_sets[1:]):
    #     print(training_set)
    #     for iM, motion in enumerate(input_motion):
    #         print(motion)

    #         # initiate dataframe
    #         data_per_layer = []

    #         for iL in range(n_layer):

    #             # select data
    #             data_current_x = np.mean(ratio_lin_log[0, iM, :, iL, :], 0)
    #             data_current_y = np.mean(ratio_trans_sust[0, iM, :, iL, :], 0)

    #             # visualize
    #             axs[iM, iTS].scatter(data_current_x, data_current_y, facecolor='lightsalmon', linewidths=0.5, edgecolor='white', s=40) #, alpha=0.5)

    # visualize correlations
    for iTS, training_set in enumerate(training_sets[1:]):
        print(training_set)
        for iM, motion in enumerate(input_motion):
            print(motion)

            for iInit in range(1):

                # initiate dataframe
                data_per_layer = []

                for iL in range(n_layer):
                # for iL in range(1):

                    # select data
                    data_current_x = ratio_lin_log[iTS+1, iM, :, iL, :].mean(0)
                    data_current_y = ratio_trans_sust[iTS+1, iM, :, iL, :].mean(0)

                    # if motion == 'static':
                    #     print(np.mean(data_current_y))
                    #     print(np.std(data_current_y))

                    # # select data
                    # data_current_x = ratio_lin_log[iTS+1, iM, iInit, iL, :]
                    # data_current_y = ratio_trans_sust[iTS+1, iM, iInit, iL, :]

                    # visualize
                    axs[iM, iTS].scatter(data_current_x, data_current_y, facecolor=color[iL], alpha=0.8, linewidths=0.5, edgecolor='white', s=30) #, alpha=0.5)

                    # # visualize
                    # axs[iM, iTS].scatter(data_current_x, data_current_y, facecolor=color[iL], marker=marker[iInit], alpha=0.8, linewidths=0.5, edgecolor='white', s=100) #, alpha=0.5)

                    # concat
                    data_current = [data_current_x, data_current_y]
                    data_per_layer.append(data_current)

                ################################################################### ALL

                # plot regression line
                x = ratio_lin_log[iTS+1, iM, :, 0, :].mean(0).tolist() + ratio_lin_log[iTS+1, iM, :, 1, :].mean(0).tolist() + ratio_lin_log[iTS+1, iM, :, 2, :].mean(0).tolist() + ratio_lin_log[iTS+1, iM, :, 3, :].mean(0).tolist()
                y = ratio_trans_sust[iTS+1, iM, :, 0, :].mean(0).tolist() + ratio_trans_sust[iTS+1, iM, :, 1, :].mean(0).tolist() + ratio_trans_sust[iTS+1, iM, :, 2, :].mean(0).tolist() + ratio_trans_sust[iTS+1, iM, :, 3, :].mean(0).tolist()

                # x = ratio_lin_log[iTS+1, iM, iInit, 0, :].tolist() + ratio_lin_log[iTS+1, iM, iInit, 1, :].tolist() + ratio_lin_log[iTS+1, iM, iInit, 2, :].tolist() + ratio_lin_log[iTS+1, iM, iInit, 3, :].tolist()
                # y = ratio_trans_sust[iTS+1, iM, iInit, 0, :].tolist() + ratio_trans_sust[iTS+1, iM, iInit, 1, :].tolist() + ratio_trans_sust[iTS+1, iM, iInit, 2, :].tolist() + ratio_trans_sust[iTS+1, iM, iInit, 3, :].tolist()

                # convert to array
                x = np.array(x)
                y = np.array(y)

                # fit linear regression
                model_LR = LinearRegression().fit(x.reshape(-1, 1), y)

                # slope and statistics
                slope = model_LR.coef_[0]
                # print("Slope:", slope)

                x_with_const = sm.add_constant(x)
                model_sm = sm.OLS(y, x_with_const).fit()
                # print(model_sm.summary())

                # predict line
                y = model_LR.intercept_ + model_LR.coef_* np.linspace(np.min(x), np.max(x), 100)
                # axs[iM, iTS].plot(np.linspace(np.min(x), np.max(x), 100), y, color='grey', lw=2, linestyle='dashed')

                # adjust axes
                axs[iM, iTS].tick_params(axis='both', labelsize=fontsize_tick)
                axs[iM, iTS].spines['top'].set_visible(False)
                axs[iM, iTS].spines['right'].set_visible(False)
                axs[iM, iTS].set_xlim(0.5, 1.3)
                if motion == 'dynamic':
                    axs[iM, iTS].set_ylim(-10, 50)
                elif motion == 'static':
                    axs[iM, iTS].set_ylim(-10, 200)
                if iTS == 0:
                    # axs[iM, iTS].set_ylabel('Ratio transient/sustained', fontsize=fontsize_label)
                    pass
                else:
                    axs[iM, iTS].set_yticks([])
                # if iM == 1:
                #     # axs[iM, iTS].set_xlabel('Ratio fit lin/log', fontsize=fontsize_label)
                #     pass
                # else:
                #     axs[iM, iTS].set_xticks([])
            
    # save figure
    plt.tight_layout()
    plt.savefig(root + test_sets + '_onepulse_dynamics_regression_all_staticmotion', dpi=600)
    plt.savefig(root + test_sets + '_onepulse_dynamics_regression_all_staticmotion.svg')
    plt.close()

def plot_regression_dataset(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, input_motion, n_layer, color, n_img, init, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 15
    fontsize_tick           = 15

    # initiate figure
    fig, axs = plt.subplots(2, 2, figsize=(6, 4.5), sharey=True, sharex=True)

    # initiate axes
    sns.despine(offset=10)

    # markers
    # marker = ['o', 'v', '^', '*', 'X']

    # plot untrained network
    for iTS, training_set in enumerate(training_sets[1:]):

        print(training_set)

        # set axes
        if iTS == 0:
            row = 0
            column = 0
        elif iTS == 1:
            row = 0
            column = 1
        elif iTS == 2:
            row = 1
            column = 0
        elif iTS == 3:
            row = 1
            column = 1

        # for iL in range(n_layer):

        #     # select data
        #     data_current_x = ratio_lin_log[0, :, iL, :].mean(0)
        #     data_current_y = ratio_trans_sust[0, :, iL, :].mean(0)

        #     # visualize
        #     axs[row, column].scatter(data_current_x, data_current_y, facecolor='lightsalmon', linewidths=0.5, edgecolor='white', s=40) #, alpha=0.5)

        # initiate dataframe
        data_per_layer = []

        for iInit in range(1):

            for iL in range(n_layer):
            # for iL in range(1):

                # select data
                data_current_x = ratio_lin_log[iTS+1, :, iL, :].mean(0) # average over inits
                data_current_y = ratio_trans_sust[iTS+1, :, iL, :].mean(0) # average over inits
                # print(data_current_x)

                # select data
                # data_current_x = ratio_lin_log[iTS, iInit, iL, :]       # average over inits
                # data_current_y = ratio_trans_sust[iTS, iInit, iL, :]    # average over inits

                # visualize
                # axs[row, column].scatter(data_current_x, data_current_y, facecolor=color[iL], alpha=1, linewidths=0.5, edgecolor='white', s=30, label=training_set) #, alpha=0.5)

                # compute metrics
                data_mean = np.mean(data_current_y)
                data_sem = np.std(data_current_y)/math.sqrt(init)

                axs[row, column].bar(iL, data_mean, color='dimgrey', zorder=-3)
                sns.stripplot(x=np.ones(len(data_current_y))*iL, y=data_current_y, jitter=True, linewidth=0.5, ax=axs[row, column], facecolor='silver', edgecolor='white', size=4, alpha=0.75, native_scale=True, zorder=-1)
                axs[row, column].plot([iL, iL], [data_mean - data_sem, data_mean + data_sem], color='black', zorder=0)
                # axs[row, column].scatter(np.ones(len(data_current_y))*iL, data_current_y, facecolor=color[iL], alpha=1, linewidths=0.5, edgecolor='white', s=30, label=training_set) #, alpha=0.5)

                # if iL == 0:
                #     axs[row, column].scatter(data_current_x, data_current_y, facecolor=color[iL], alpha=0.4+0.15*iL, linewidths=0.5, edgecolor='white', s=30, label=training_set) #, alpha=0.5)
                # else:
                #     axs[row, column].scatter(data_current_x, data_current_y, facecolor=color[iL], alpha=0.4+0.15*iL, linewidths=0.5, edgecolor='white', s=30) #, alpha=0.5)

                # visualize
                # axs[row, column].scatter(data_current_x, data_current_y, facecolor=color[iL], marker=marker[iInit], alpha=0.8, linewidths=0.5, edgecolor='white', s=100) #, alpha=0.5)

                # # concat
                # data_current = [data_current_x, data_current_y]
                # data_per_layer.append(data_current)

                # # plot spread across network initializations
                # for i in range(2):
                #     for iImg in range(n_img):

                #         # select data
                #         data_current_x = ratio_lin_log[iTS+1, :, iL, iImg]
                #         data_current_y = ratio_trans_sust[iTS+1, :, iL, iImg]

                #         # compute spread
                #         if i == 0:

                #             # compute
                #             mean = np.mean(data_current_x)
                #             sem = np.std(data_current_x)/math.sqrt(init)
                #             # print(sem)

                #             # visualize
                #             axs[row, column].plot([mean - sem, mean + sem], [np.mean(data_current_y), np.mean(data_current_y)], lw=0.5, color=color[iL], alpha=0.5, zorder=-100)

                #         else:

                #             # compute
                #             mean = np.mean(data_current_y)
                #             sem = np.std(data_current_y)/math.sqrt(init)
                #             # print(training_set, iL, sem)

                #             # visualize
                #             axs[row, column].plot([np.mean(data_current_x), np.mean(data_current_x)], [mean - sem, mean + sem], lw=0.5, color=color[iL], alpha=0.5, zorder=-100)


            ################################################################### ALL

            # plot regression line
            x = ratio_lin_log[iTS+1, :, 0, :].mean(0).tolist() + ratio_lin_log[iTS+1, :, 1, :].mean(0).tolist() + ratio_lin_log[iTS+1, :, 2, :].mean(0).tolist() + ratio_lin_log[iTS+1, :, 3, :].mean(0).tolist()
            y = ratio_trans_sust[iTS+1, :, 0, :].mean(0).tolist() + ratio_trans_sust[iTS+1, :, 1, :].mean(0).tolist() + ratio_trans_sust[iTS+1, :, 2, :].mean(0).tolist() + ratio_trans_sust[iTS+1, :, 3, :].mean(0).tolist()

            # x = ratio_lin_log[iTS+1, iM, iInit, 0, :].tolist() + ratio_lin_log[iTS+1, iM, iInit, 1, :].tolist() + ratio_lin_log[iTS+1, iM, iInit, 2, :].tolist() + ratio_lin_log[iTS+1, iM, iInit, 3, :].tolist()
            # y = ratio_trans_sust[iTS+1, iM, iInit, 0, :].tolist() + ratio_trans_sust[iTS+1, iM, iInit, 1, :].tolist() + ratio_trans_sust[iTS+1, iM, iInit, 2, :].tolist() + ratio_trans_sust[iTS+1, iM, iInit, 3, :].tolist()

            # convert to array
            x = np.array(x)
            y = np.array(y)

            # fit linear regression
            model_LR = LinearRegression().fit(x.reshape(-1, 1), y)

            # slope and statistics
            slope = model_LR.coef_[0]
            # print("Slope:", slope)

            x_with_const = sm.add_constant(x)
            model_sm = sm.OLS(y, x_with_const).fit()
            print(model_sm.summary())

            # predict line
            y = model_LR.intercept_ + model_LR.coef_* np.linspace(np.min(x), np.max(x), 100)
            # axs[row, column].plot(np.linspace(np.min(x), np.max(x), 100), y, color='grey', lw=2, linestyle='dashed')

            # adjust axes
            axs[row, column].tick_params(axis='both', labelsize=fontsize_tick)
            axs[row, column].spines['top'].set_visible(False)
            axs[row, column].spines['right'].set_visible(False)
            axs[row, column].set_xticks(np.arange(n_layer))
            if (iL == 2) | (iL == 3):
                axs[row, column].set_xticklabels(['E1', 'E2', 'E3', 'E4'], fontsize=fontsize_label, rotation=45)
            else:
                axs[row, column].set_xticklabels(['', '', '', ''])

            # axs[row, column].set_xlim(0.5, 1.3)
            # axs[row, column].set_ylim(-10, 200)
            
    # save figure
    # plt.legend()
    plt.tight_layout()
    plt.savefig(root + 'onepulse_dynamics_regression_dataset', dpi=600)
    plt.savefig(root + 'onepulse_dynamics_regression_dataset.svg')
    plt.close()

def plot_regression_dataset_all(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, input_motion, n_layer, color, n_img, init, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 15
    fontsize_tick           = 15

    # initiate figure
    fig, axs = plt.subplots(1, 4, figsize=(7, 3), sharey=True, sharex=True)

    # initiate axes
    sns.despine(offset=10)

    # set data_mean
    neural_data_mean = 1.2125099187860158

    # plot untrained network
    for iTS, training_set in enumerate(training_sets[1:]):

        # initiate dataframe
        data_per_layer = []

        for iInit in range(1):

            for iL in range(n_layer):
            # for iL in range(1):

                # select data
                data_current_x = ratio_lin_log[iTS+1, :, iL, :].mean(0) # average over inits
                data_current_y = ratio_lin_log[iTS+1, :, iL, :].mean(0) # average over inits
                # print(data_current_x)

                print(training_set, 'layer', iL+1)
                results = stats.ttest_1samp(data_current_y, 1)
                if results[1] < 0.05:
                    print(results)

                # plot data mean
                axs[iTS].scatter(iL, neural_data_mean, color='black', marker='_', zorder=-3)

                # compute metrics
                data_mean = np.mean(data_current_y)
                data_sem = np.std(data_current_y)/math.sqrt(init)

                # axs[row, column].bar(iL, data_mean, color='dimgrey', zorder=-3)
                sns.stripplot(x=np.ones(len(data_current_y))*iL, y=data_current_y, jitter=True, linewidth=0.5, ax=axs[iTS], facecolor=color[iL], edgecolor='white', size=4, alpha=0.5, native_scale=True, zorder=-1)
                axs[iTS].scatter(iL, data_mean, facecolor=color[iL], edgecolor='white', zorder=0, linewidths=0.5,)
                # axs[iTS].plot([iL, iL], [data_mean - data_sem, data_mean + data_sem], color='dimgrey', zorder=0)
                # axs[row, column].scatter(np.ones(len(data_current_y))*iL, data_current_y, facecolor=color[iL], alpha=1, linewidths=0.5, edgecolor='white', s=30, label=training_set) #, alpha=0.5)

            # adjust axes
            axs[iTS].tick_params(axis='both', labelsize=fontsize_tick)
            axs[iTS].spines['top'].set_visible(False)
            axs[iTS].spines['right'].set_visible(False)
            axs[iTS].set_xticks(np.arange(n_layer))
            # if (iL == 2) | (iL == 3):
            #     axs[iTS].set_xticklabels(['E1', 'E2', 'E3', 'E4'], fontsize=fontsize_label, rotation=45)
            # else:
            axs[iTS].set_xticklabels(['', '', '', ''])

            # axs[row, column].set_xlim(0.5, 1.3)
            # axs[row, column].set_ylim(-10, 200)
            
    # save figure
    # plt.legend()
    plt.tight_layout()
    plt.savefig(root + 'onepulse_dynamics_regression_dataset', dpi=600)
    plt.savefig(root + 'onepulse_dynamics_regression_dataset.svg')
    plt.close()

def plot_regression_all_datasets_fps(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color, n_img, init, root, datasets):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 15
    fontsize_label          = 18
    fontsize_tick           = 16

    # initiate figure
    fig, axs = plt.subplots(4, 1, figsize=(4, 10))#, sharex=True, sharey=True)

    # initiate axes
    sns.despine(offset=10)

    # colors
    color = ['khaki', 'sandybrown', 'tomato']

    # set data_mean
    neural_data_mean = 1.2125099187860158

    # visualize
    offset_iL = [0, 5, 10, 15]
    for iD, dataset in enumerate(datasets):
        for iL in range(n_layer):

            # plot 
            for i in range(3):

                # select data
                data_current_y = ratio_lin_log[iD, i, :, iL, :].mean(0)

                # compute metrics
                data_mean = np.mean(data_current_y)
                data_sem = np.std(data_current_y)/math.sqrt(init)

                # plot data mean
                axs[iD].scatter(offset_iL[iL]+i, neural_data_mean, color='black', marker='_', zorder=-3)

                # axs[row, column].bar(iL, data_mean, color='dimgrey', zorder=-3)
                sns.stripplot(x=np.ones(len(data_current_y))*(offset_iL[iL]+i), y=data_current_y, jitter=True, linewidth=0.5, ax=axs[iD], facecolor=color[i], edgecolor='white', size=4, alpha=0.5, native_scale=True, zorder=-1)
                axs[iD].scatter(offset_iL[iL]+i, data_mean, color=color[i], zorder=0)
                axs[iD].plot([offset_iL[iL]+i, offset_iL[iL]+i], [data_mean - data_sem, data_mean + data_sem], color='dimgrey', zorder=0, lw=0.75)
                # axs[row, column].scatter(np.ones(len(data_current_y))*iL, data_current_y, facecolor=color[iL], alpha=1, linewidths=0.5, edgecolor='white', s=30, label=training_set) #, alpha=0.5)

            # adjust axes
            axs[iD].tick_params(axis='both', labelsize=fontsize_tick)
            axs[iD].spines['top'].set_visible(False)
            axs[iD].spines['right'].set_visible(False)
            axs[iD].set_xticks([offset_iL[0]+1, offset_iL[1]+1, offset_iL[2]+1, offset_iL[3]+1])
            axs[iD].set_xticklabels([' ', ' ', ' ', ' '])

            # compute statistics
            print(dataset, ', layer ', iL + 1)
            results = f_oneway(ratio_lin_log[iD, 0, :, iL, :].flatten(), ratio_lin_log[iD, 1, :, iL, :].flatten(), ratio_lin_log[iD, 2, :, iL, :].flatten())
            print(results)
            results = tukey_hsd(ratio_lin_log[iD, 0, :, iL, :].flatten(), ratio_lin_log[iD, 1, :, iL, :].flatten(), ratio_lin_log[iD, 2, :, iL, :].flatten())
            print(results)
                
    # save figure
    plt.tight_layout()
    plt.savefig(root + 'onepulse_dynamics_regression_fps', dpi=600)
    plt.savefig(root + 'onepulse_dynamics_regression_fps.svg')
    plt.close()

def plot_regression_all_fps(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color, n_img, init, root, dataset):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 15
    fontsize_label          = 18
    fontsize_tick           = 16

    # initiate figure
    fig = plt.figure(figsize=(4, 2.5))#, sharex=True, sharey=True)
    ax1 = plt.gca()

    # initiate axes
    sns.despine(offset=10)

    # colors
    color = ['khaki', 'sandybrown', 'tomato']

    # visualize
    offset_iL = [0, 5, 10, 15]
    for iL in range(n_layer):

        # plot 
        for i in range(3):

            # select data
            # data_current_x = ratio_lin_log[i, :, iL, :].mean(1)
            data_current_y = ratio_trans_sust[i, :, iL, :].mean(0)

            # compute metrics
            data_mean = np.mean(data_current_y)
            data_sem = np.std(data_current_y)/math.sqrt(init)

            # axs[row, column].bar(iL, data_mean, color='dimgrey', zorder=-3)
            sns.stripplot(x=np.ones(len(data_current_y))*(offset_iL[iL]+i), y=data_current_y, jitter=True, linewidth=0.5, ax=ax1, facecolor=color[i], edgecolor='white', size=4, alpha=0.5, native_scale=True, zorder=-1)
            ax1.scatter(offset_iL[iL]+i, data_mean, color=color[i], zorder=0)
            ax1.plot([offset_iL[iL]+i, offset_iL[iL]+i], [data_mean - data_sem, data_mean + data_sem], color='dimgrey', zorder=0, lw=0.75)
            # axs[row, column].scatter(np.ones(len(data_current_y))*iL, data_current_y, facecolor=color[iL], alpha=1, linewidths=0.5, edgecolor='white', s=30, label=training_set) #, alpha=0.5)

        # adjust axes
        ax1.tick_params(axis='both', labelsize=fontsize_tick)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xticks([offset_iL[0]+1, offset_iL[1]+1, offset_iL[2]+1, offset_iL[3]+1])
        ax1.set_xticklabels([' ', ' ', ' ', ' '])

        # compute statistics
        results = f_oneway(ratio_trans_sust[0, :, iL, :].flatten(), ratio_trans_sust[1, :, iL, :].flatten(), ratio_trans_sust[2, :, iL, :].flatten())
        print(results)
        results = tukey_hsd(ratio_trans_sust[0, :, iL, :].flatten(), ratio_trans_sust[1, :, iL, :].flatten(), ratio_trans_sust[2, :, iL, :].flatten())
        print(results)
        # ax1.set_xticks([offset_iI[0], offset_iI[0]+1, offset_iI[0]+2, offset_iI[0]+3, offset_iI[1], offset_iI[1]+1, offset_iI[1]+2, offset_iI[1]+3])
        # ax1.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], fontsize=fontsize_label)
            
    # save figure
    # plt.tight_layout()
    plt.savefig(root + 'onepulse_dynamics_regression_fps_' + dataset, dpi=600)
    plt.savefig(root + 'onepulse_dynamics_regression_fps_' + dataset + '.svg')
    plt.close()


def plot_regression_all_datasets_loss(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color, n_img, init, root, datasets):


    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 15
    fontsize_label          = 18
    fontsize_tick           = 16

    # initiate figure
    fig, axs = plt.subplots(4, 1, figsize=(4, 10))#, sharex=True, sharey=True)

    # initiate axes
    sns.despine(offset=10)

    # colors
    color = ['#125A56', '#238F9D']

    # set data_mean
    neural_data_mean = 1.2125099187860158

    # set maximal values
    y_min = [0.75, 0.75, 0.75, 0.8]
    y_max = [1.7, 1.5, 1.7, 2]

    # visualize
    offset_iL = [0, 5, 10, 15]
    for iD, dataset in enumerate(datasets):
        for iL in range(n_layer):

            # plot 
            for i in range(2):

                # select data
                data_current_y = ratio_lin_log[iD, i, :, iL, :].mean(0)

                # compute metrics
                data_mean = np.mean(data_current_y)
                data_sem = np.std(data_current_y)/math.sqrt(init)

                # plot data mean
                axs[iD].scatter(offset_iL[iL]+i, neural_data_mean, color='black', marker='_', zorder=-3)

                # axs[row, column].bar(iL, data_mean, color='dimgrey', zorder=-3)
                sns.stripplot(x=np.ones(len(data_current_y))*(offset_iL[iL]+i), y=data_current_y, jitter=True, linewidth=0.5, ax=axs[iD], facecolor=color[i], edgecolor='white', size=4, alpha=0.5, native_scale=True, zorder=-1)
                axs[iD].scatter(offset_iL[iL]+i, data_mean, color=color[i], zorder=0)
                axs[iD].plot([offset_iL[iL]+i, offset_iL[iL]+i], [data_mean - data_sem, data_mean + data_sem], color='dimgrey', zorder=0, lw=0.75)
                # axs[row, column].scatter(np.ones(len(data_current_y))*iL, data_current_y, facecolor=color[iL], alpha=1, linewidths=0.5, edgecolor='white', s=30, label=training_set) #, alpha=0.5)

            # adjust axes
            axs[iD].tick_params(axis='both', labelsize=fontsize_tick)
            axs[iD].spines['top'].set_visible(False)
            axs[iD].spines['right'].set_visible(False)
            axs[iD].set_xticks([offset_iL[0]+1, offset_iL[1]+0.5, offset_iL[2]+0.5, offset_iL[3]+0.5])
            axs[iD].set_xticklabels([' ', ' ', ' ', ' '])
            axs[iD].set_ylim(y_min[iD], y_max[iD])

        # compute statistics
        for i in range(2):
            print(dataset, i+ 1)
            results = f_oneway(ratio_lin_log[iD, i, :, 0, :].flatten(), ratio_lin_log[iD, i, :, 1, :].flatten(), ratio_lin_log[iD, i, :, 2, :].flatten(), ratio_lin_log[iD, i, :, 3, :].flatten())
            print(results)
            results = tukey_hsd(ratio_lin_log[iD, i, :, 0, :].flatten(), ratio_lin_log[iD, i, :, 1, :].flatten(), ratio_lin_log[iD, i, :, 2, :].flatten(), ratio_lin_log[iD, i, :, 3, :].flatten())
            print(results)
            # ax1.set_xticks([offset_iI[0], offset_iI[0]+1, offset_iI[0]+2, offset_iI[0]+3, offset_iI[1], offset_iI[1]+1, offset_iI[1]+2, offset_iI[1]+3])
            # ax1.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], fontsize=fontsize_label)

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'onepulse_dynamics_regression_loss', dpi=600)
    plt.savefig(root + 'onepulse_dynamics_regression_loss.svg')
    plt.close()

def plot_regression_all_loss(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color, n_img, init, root, dataset):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 15
    fontsize_label          = 18
    fontsize_tick           = 16

    # initiate figure
    fig = plt.figure(figsize=(4, 2.5))#, sharex=True, sharey=True)
    ax1 = plt.gca()

    # initiate axes
    sns.despine(offset=10)

    # colors
    color = ['#125A56', '#238F9D']

    # visualize
    offset_iL = [0, 5, 10, 15]
    for iL in range(n_layer):

        # plot 
        for i in range(2):

            # select data
            # data_current_x = ratio_lin_log[i, :, iL, :].mean(1)
            data_current_y = ratio_trans_sust[i, :, iL, :].mean(0)

            # compute metrics
            data_mean = np.mean(data_current_y)
            data_sem = np.std(data_current_y)/math.sqrt(init)

            # axs[row, column].bar(iL, data_mean, color='dimgrey', zorder=-3)
            sns.stripplot(x=np.ones(len(data_current_y))*(offset_iL[iL]+i), y=data_current_y, jitter=True, linewidth=0.5, ax=ax1, facecolor=color[i], edgecolor='white', size=4, alpha=0.5, native_scale=True, zorder=-1)
            ax1.scatter(offset_iL[iL]+i, data_mean, color=color[i], zorder=0)
            ax1.plot([offset_iL[iL]+i, offset_iL[iL]+i], [data_mean - data_sem, data_mean + data_sem], color='dimgrey', zorder=0, lw=0.75)
            # axs[row, column].scatter(np.ones(len(data_current_y))*iL, data_current_y, facecolor=color[iL], alpha=1, linewidths=0.5, edgecolor='white', s=30, label=training_set) #, alpha=0.5)

        # adjust axes
        ax1.tick_params(axis='both', labelsize=fontsize_tick)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xticks([offset_iL[0]+1, offset_iL[1]+0.5, offset_iL[2]+0.5, offset_iL[3]+0.5])
        ax1.set_xticklabels([' ', ' ', ' ', ' '])

        # compute statistics
        print('L', i+1)
        results = f_oneway(ratio_trans_sust[0, :, iL, :].flatten(), ratio_trans_sust[1, :, iL, :].flatten())
        print(results)
        results = tukey_hsd(ratio_trans_sust[0, :, iL, :].flatten(), ratio_trans_sust[1, :, iL, :].flatten())
        print(results)
        # ax1.set_xticks([offset_iI[0], offset_iI[0]+1, offset_iI[0]+2, offset_iI[0]+3, offset_iI[1], offset_iI[1]+1, offset_iI[1]+2, offset_iI[1]+3])
        # ax1.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], fontsize=fontsize_label)

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'onepulse_dynamics_regression_loss_' + dataset, dpi=600)
    plt.savefig(root + 'onepulse_dynamics_regression_loss_' + dataset + '.svg')
    plt.close()

def plot_regression_all_loss_old(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color, n_img, init, root, dataset):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 15
    fontsize_label          = 18
    fontsize_tick           = 16

    # initiate figure
    fig = plt.figure(figsize=(4, 2.5))#, sharex=True, sharey=True)
    ax1 = plt.gca()
    ax2 = ax1.twinx() 

    # initiate axes
    sns.despine(offset=10)

    # color plot
    # color = ['#125A56', '#238F9D']
    color = ['darkkhaki', 'darkseagreen']

    # initiate dataframe
    data = np.zeros((2, init, n_img))

    # visualize
    offset_iI = [0, 6]
    for iL in range(n_layer):
        for iInit in range(init):

            # retrieve data
            data[0, iInit, :] = abs(ratio_lin_log[1, iInit, iL, :] - ratio_lin_log[0, iInit, iL, :])
            data[1, iInit, :] = abs(ratio_trans_sust[1, iInit, iL, :] - ratio_trans_sust[0, iInit, iL, :])

        # plot 
        for i in range(2):

            # select data
            data_current = data[i, :, :].mean(1)

            # compute
            mean = np.mean(data_current)
            sem = np.std(data_current)/math.sqrt(init)

            # visualize
            if i == 0:
                ax1.bar(offset_iI[i]+iL, mean, color=color[i], edgecolor='white', width=1)
                ax1.plot([offset_iI[i]+iL, offset_iI[i]+iL], [mean - sem, mean + sem], color='black', lw=0.75)
                ax1.tick_params(axis='y', labelsize=fontsize_tick, labelcolor=color[i])
            else:
                ax2.bar(offset_iI[i]+iL, mean, color=color[i], edgecolor='white', width=1)
                ax2.plot([offset_iI[i]+iL, offset_iI[i]+iL], [mean - sem, mean + sem], color='black', lw=0.75)
                ax2.tick_params(axis='y', labelsize=fontsize_tick, labelcolor=color[i])
        
        # adjust axes
        ax1.tick_params(axis='x', labelsize=fontsize_tick)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(True)
        ax1.set_xticks([offset_iI[0], offset_iI[0]+1, offset_iI[0]+2, offset_iI[0]+3, offset_iI[1], offset_iI[1]+1, offset_iI[1]+2, offset_iI[1]+3])
        ax1.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], fontsize=fontsize_label)
            
    # save figure
    plt.tight_layout()
    plt.savefig(root + 'onepulse_dynamics_regression_loss_diff_dataset', dpi=600)
    plt.savefig(root + 'onepulse_dynamics_regression_loss_diff_' + dataset + '.svg')
    plt.close()

def plot_regression_CEandSC_L1(train_sets, ratio_lin_log, ratio_trans_sust, CEandSC_values, n_layer, color, n_img, init, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 20
    fontsize_tick           = 15

    # initiate figure
    fig, axs = plt.subplots(2, 1, figsize=(4, 6))
    
    sns.despine(offset=10)

    # set title
    # fig.suptitle(model, fontsize=fontsize_title)

    # summary statistics
    summary_statistics  = ['CE', 'SC']

    # visualize correlations
    for iT, train_set, in enumerate(train_sets):
        for iSumStat, SumStat in enumerate(summary_statistics):

            # select data
            data_current_x = CEandSC_values[:, iSumStat]
            data_current_y = ratio_trans_sust[iT, :, 0, :].mean(0)

            # visualize
            axs[iSumStat].scatter(data_current_x, data_current_y, color='white', facecolor=color[iT], s=100, linewidth=1.5)#, alpha=0.7)

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
            print(model_sm.summary())

            # predict line
            y = model.intercept_ + model.coef_*np.linspace(np.min(x), np.max(x), 100)
            sc = axs[iSumStat].plot(np.linspace(np.min(x), np.max(x), 100), y, color=color[iT], lw=3, linestyle='dashed')

            # adjust axes
            axs[iSumStat].tick_params(axis='both', labelsize=fontsize_tick)
            axs[iSumStat].spines['top'].set_visible(False)
            axs[iSumStat].spines['right'].set_visible(False)
            axs[iSumStat].set_xlabel(SumStat, fontsize=fontsize_label)

    # save figure
    fig.align_ylabels()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    plt.savefig(root + 'onepulse_dynamics_regression_CEandSC_L1.png', dpi=600)
    plt.savefig(root + 'onepulse_dynamics_regression_CEandSC_L1.svg')
    plt.close()


def plot_regression_CEandSC_L1_all(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, input_motion, CEandSC_values, n_layer, color, n_img, init, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 20
    fontsize_tick           = 15

    # initiate figure
    fig, axs = plt.subplots(1, (len(training_sets)-1)*2, figsize=(20, 3))

    # spread axes
    sns.despine(offset=10)

    # summary statistics
    summary_statistics  = ['CE', 'SC']

    # visualize correlations
    for iTS, training_set in enumerate(training_sets[1:]):
        for iSumStat, SumStat in enumerate(summary_statistics):

            # select data
            data_current_x = CEandSC_values[:, iSumStat]
            data_current_y = ratio_trans_sust[iTS+1, :, 0, :].mean(0)

            # visualize
            axs[iTS*2+iSumStat].scatter(data_current_x, data_current_y, edgecolor='white', facecolor='grey', s=80, linewidth=1.5, alpha=1)

            # plot spread across network initializations
            for iImg in range(n_img):

                # select data
                data_current_x_per_img = CEandSC_values[iImg, iSumStat]
                data_current_y_per_img = ratio_trans_sust[iTS+1, :, 0, iImg]

                # compute
                mean = np.mean(data_current_y_per_img)
                sem = np.std(data_current_y_per_img)/math.sqrt(init)
                # print(sem)

                # visualize
                axs[iTS*2+iSumStat].plot([data_current_x_per_img, data_current_x_per_img], [mean - sem, mean + sem], lw=1, color='grey', alpha=1, zorder=-100)

            # select data
            x = data_current_x
            y = data_current_y
            
            # fit linear regression
            model = LinearRegression().fit(x.reshape(-1, 1), y)

            # slope and statistics
            slope = model.coef_[0]
            print(training_set, SumStat)
            # print("Slope:", slope)

            x_with_const = sm.add_constant(x)
            model_sm = sm.OLS(y, x_with_const).fit()
            print(model_sm.summary())

            # predict line
            y = model.intercept_ + model.coef_*np.linspace(np.min(x), np.max(x), 100)
            sc = axs[iTS*2+iSumStat].plot(np.linspace(np.min(x), np.max(x), 100), y, color='crimson', lw=3, linestyle='dashed')

            # adjust axes
            axs[iTS*2+iSumStat].tick_params(axis='both', labelsize=fontsize_tick)
            axs[iTS*2+iSumStat].spines['top'].set_visible(False)
            axs[iTS*2+iSumStat].spines['right'].set_visible(False)
            # axs[iTS*2+iSumStat].set_xlabel(SumStat, fontsize=fontsize_label)

    # save figure
    fig.align_ylabels()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    plt.savefig(root + 'onepulse_dynamics_regression_dataset_CEandSC_L1.png', dpi=600)
    plt.savefig(root + 'onepulse_dynamics_regression_dataset_CEandSC_L1.svg')
    plt.close()


# def stats_lin_log(n_layer, dynamics_fit):

#     alpha = 0.05

#     for iL in range(n_layer):

#         # print progress
#         print('Layer', iL+1)

#         # ttest
#         sample1 = dynamics_fit[iL, :, 0]
#         sample2 = dynamics_fit[iL, :, 1]
#         p = stats.ttest_rel(sample1, sample2)[1]
#         if p < alpha:
#             print('Linear vs. log fit', p, ' SIGNIFICANT')
#         else:
#             print('Linear vs. log fit', p)

#         print('\n')

# def stats_regression(ratio_lin_log, ratio_trans_sust):

#     # oneway anova
#     result = f_oneway(ratio_lin_log[0, :], ratio_lin_log[1, :], ratio_lin_log[2, :])

#     # convert to list
#     x = ratio_lin_log[0, :].tolist() + ratio_lin_log[1, :].tolist() + ratio_lin_log[2, :].tolist() + ratio_lin_log[3, :].tolist()
#     y = ratio_trans_sust[0, :].tolist() + ratio_trans_sust[1, :].tolist() + ratio_trans_sust[2, :].tolist() + ratio_trans_sust[3, :].tolist()
 
#     # convert to array
#     x = np.array(x)
#     y = np.array(y)

#     # fit linear regression
#     model = LinearRegression().fit(x.reshape(-1, 1), y)

#     # slope and statistics
#     slope = model.coef_[0]
#     print("Slope:", slope)

#     x_with_const = sm.add_constant(x)
#     model_sm = sm.OLS(y, x_with_const).fit()
#     print(model_sm.summary())