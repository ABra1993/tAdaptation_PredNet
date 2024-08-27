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

def plot_regression_dataset_Fig6E(ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color, n_img, init, root):

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

        # initiate dataframe
        data_per_layer = []

        for iInit in range(1):

            for iL in range(n_layer):
            # for iL in range(1):

                # select data
                data_current_x = ratio_lin_log[iTS+1, :, iL, :].mean(0) # average over inits
                data_current_y = ratio_trans_sust[iTS+1, :, iL, :].mean(0) # average over inits
                # print(data_current_x)

                # visualize
                axs[row, column].scatter(data_current_x, data_current_y, facecolor=color[iL], alpha=1, linewidths=0.5, edgecolor='white', s=30, label=training_set) #, alpha=0.5)

                # concat
                data_current = [data_current_x, data_current_y]
                data_per_layer.append(data_current)

                # plot spread across network initializations
                for i in range(2):
                    for iImg in range(n_img):

                        # select data
                        data_current_x = ratio_lin_log[iTS+1, :, iL, iImg]
                        data_current_y = ratio_trans_sust[iTS+1, :, iL, iImg]

                        # compute spread
                        if i == 0:

                            # compute
                            mean = np.mean(data_current_x)
                            sem = np.std(data_current_x)/math.sqrt(init)
                            # print(sem)

                            # visualize
                            axs[row, column].plot([mean - sem, mean + sem], [np.mean(data_current_y), np.mean(data_current_y)], lw=0.5, color=color[iL], alpha=0.5, zorder=-100)

                        else:

                            # compute
                            mean = np.mean(data_current_y)
                            sem = np.std(data_current_y)/math.sqrt(init)
                            # print(training_set, iL, sem)

                            # visualize
                            axs[row, column].plot([np.mean(data_current_x), np.mean(data_current_x)], [mean - sem, mean + sem], lw=0.5, color=color[iL], alpha=0.5, zorder=-100)


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
            axs[row, column].plot(np.linspace(np.min(x), np.max(x), 100), y, color='grey', lw=2, linestyle='dashed')

            # adjust axes
            axs[row, column].tick_params(axis='both', labelsize=fontsize_tick)
            axs[row, column].spines['top'].set_visible(False)
            axs[row, column].spines['right'].set_visible(False)
            # axs[row, column].set_xlim(0.5, 1.3)
            # axs[row, column].set_ylim(-10, 200)
            
    # save figure
    # plt.legend()
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig6E.png', dpi=600)
    plt.savefig(root + 'visualization/Fig6E.svg')
    plt.close()

def plot_regression_dataset_Fig6H(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, input_motion, CEandSC_values, n_layer, color, n_img, init, root):

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
    plt.savefig(root + 'visualization/Fig6H.png', dpi=600)
    plt.savefig(root + 'visualization/Fig6H.svg')
    plt.close()

def plot_regression_all_fps_Fig7B(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color, n_img, init, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 15
    fontsize_label          = 18
    fontsize_tick           = 16

    # initiate figure
    fig, axs = plt.subplots(1, n_layer, figsize=(8, 2.5))#, sharex=True, sharey=True)

    # initiate axes
    sns.despine(offset=10)

    # colors
    color = ['khaki', 'sandybrown', 'tomato']

    # visualize correlations
    for iTS, training_set in enumerate(training_sets):
        print(training_set)
        for iInit in range(1):

            # initiate dataframe
            data_per_layer = []

            for iL in range(n_layer):
            # for iL in range(1):

                # select data
                data_current_x = ratio_lin_log[iTS, :, iL, :].mean(0) # average over inits
                data_current_y = ratio_trans_sust[iTS, :, iL, :].mean(0) # average over inits

                # visualize
                axs[iL].scatter(data_current_x, data_current_y, facecolor=color[iTS], alpha=0.8, linewidths=0.5, edgecolor='white', s=30) #, alpha=0.5)

                # plot spread across network initializations
                for i in range(2):
                    for iImg in range(n_img):

                        # select data
                        data_current_x = ratio_lin_log[iTS, :, iL, iImg]
                        data_current_y = ratio_trans_sust[iTS, :, iL, iImg]

                        # compute spread
                        if i == 0:

                            # compute
                            mean = np.mean(data_current_x)
                            sem = np.std(data_current_x)/math.sqrt(init)
                            # print(sem)

                            # visualize
                            axs[iL].plot([mean - sem, mean + sem], [np.mean(data_current_y), np.mean(data_current_y)], lw=0.5, color=color[iTS], alpha=0.5, zorder=-100)

                        else:

                            # compute
                            mean = np.mean(data_current_y)
                            sem = np.std(data_current_y)/math.sqrt(init)
                            # print(training_set, iL, sem)

                            # visualize
                            axs[iL].plot([np.mean(data_current_x), np.mean(data_current_x)], [mean - sem, mean + sem], lw=0.5, color=color[iTS], alpha=0.5, zorder=-100)


                # # visualize
                # axs[iM, iTS].scatter(data_current_x, data_current_y, facecolor=color[iL], marker=marker[iInit], alpha=0.8, linewidths=0.5, edgecolor='white', s=100) #, alpha=0.5)

                # concat
                data_current = [data_current_x, data_current_y]
                data_per_layer.append(data_current)

                ################################################################### ALL

                axs[iL].tick_params(axis='both', labelsize=fontsize_tick)
                axs[iL].spines['top'].set_visible(False)
                axs[iL].spines['right'].set_visible(False)
                if iL == 0:
                    axs[iL].set_xticks([1, 1.25])
                else:
                    axs[iL].set_xticks([1.15, 1.25])
            
    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig7B.png', dpi=600)
    plt.savefig(root + 'visualization/Fig7B.svg')
    plt.close()

def plot_regression_all_fps_Fig7C(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color, n_img, init, root):

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
            data_current = ratio_trans_sust[i, :, iL, :].mean(1)
            # data_current = ratio_lin_log[i, :, iL, :].mean(1)

            # compute
            mean = np.mean(data_current)
            sem = np.std(data_current)/math.sqrt(init)

            # visualize
            ax1.bar(offset_iL[iL]+i, mean, color=color[i], edgecolor='white', width=1)
            ax1.plot([offset_iL[iL]+i, offset_iL[iL]+i], [mean - sem, mean + sem], color='black', lw=0.75)

        # adjust axes
        ax1.tick_params(axis='both', labelsize=fontsize_tick)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # # compute statistics
        # results = f_oneway(ratio_trans_sust[0, :, iL, :].flatten(), ratio_trans_sust[1, :, iL, :].flatten(), ratio_trans_sust[2, :, iL, :].flatten())
        # print(results)
        # results = tukey_hsd(ratio_trans_sust[0, :, iL, :].flatten(), ratio_trans_sust[1, :, iL, :].flatten(), ratio_trans_sust[2, :, iL, :].flatten())
        # print(results)

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig7C.png', dpi=600)
    plt.savefig(root + 'visualization/Fig7C.svg')
    plt.close()

def plot_regression_all_loss_Fig7E(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color, n_img, init, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 15
    fontsize_label          = 18
    fontsize_tick           = 16

    # initiate figure
    fig, axs = plt.subplots(1, n_layer, figsize=(8, 2.5))#, sharex=True, sharey=True)

    # initiate axes
    sns.despine(offset=10)

    # markers
    color = ['#125A56', '#238F9D']

    # visualize correlations
    for iTS, training_set in enumerate(training_sets):
        print(training_set)

        for iInit in range(1):

            # initiate dataframe
            data_per_layer = []

            for iL in range(n_layer):
            # for iL in range(1):

                # select data
                data_current_x = ratio_lin_log[iTS, :, iL, :].mean(0) # average over inits
                data_current_y = ratio_trans_sust[iTS, :, iL, :].mean(0) # average over inits

                # visualize
                axs[iL].scatter(data_current_x, data_current_y, facecolor=color[iTS], alpha=0.8, linewidths=0.5, edgecolor='white', s=30) #, alpha=0.5)

                # plot spread across network initializations
                for i in range(2):
                    for iImg in range(n_img):

                        # select data
                        data_current_x = ratio_lin_log[iTS, :, iL, iImg]
                        data_current_y = ratio_trans_sust[iTS, :, iL, iImg]

                        # compute spread
                        if i == 0:

                            # compute
                            mean = np.mean(data_current_x)
                            sem = np.std(data_current_x)/math.sqrt(init)
                            # print(sem)

                            # visualize
                            axs[iL].plot([mean - sem, mean + sem], [np.mean(data_current_y), np.mean(data_current_y)], lw=0.5, color=color[iTS], alpha=0.5, zorder=-100)

                        else:

                            # compute
                            mean = np.mean(data_current_y)
                            sem = np.std(data_current_y)/math.sqrt(init)
                            # print(training_set, iL, sem)

                            # visualize
                            axs[iL].plot([np.mean(data_current_x), np.mean(data_current_x)], [mean - sem, mean + sem], lw=0.5, color=color[iTS], alpha=0.5, zorder=-100)


                # # visualize
                # axs[iM, iTS].scatter(data_current_x, data_current_y, facecolor=color[iL], marker=marker[iInit], alpha=0.8, linewidths=0.5, edgecolor='white', s=100) #, alpha=0.5)

                # concat
                data_current = [data_current_x, data_current_y]
                data_per_layer.append(data_current)

                ################################################################### ALL

                axs[iL].tick_params(axis='both', labelsize=fontsize_tick)
                axs[iL].spines['top'].set_visible(False)
                axs[iL].spines['right'].set_visible(False)

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig7E.png', dpi=600)
    plt.savefig(root + 'visualization/Fig7E.svg')
    plt.close()

def plot_regression_all_loss_Fig7F(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color, n_img, init, root):

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
    plt.savefig(root + 'visualization/Fig7F.png', dpi=600)
    plt.savefig(root + 'visualization/Fig7F.svg')
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
