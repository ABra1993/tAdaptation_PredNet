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

def plot_broadband(data, start, tempCond, n_layer, n_img, range_trans, range_sust, trained, population, dataset, output_mode, color_tempCond, color_layer, root):

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

    # PLOT AVERAGE
    lbls = []
    for iL in range(n_layer):

        # set label
        if population != None:
            lbl = output_mode + str(iL+1) + population
        else:
            lbl = output_mode + str(iL+1)
        lbls.append(lbl)

        for iC in range(len(tempCond)):

            # select data
            data_current = data[:, iL, iC, :, start_plot:].mean(0)
            # print(data_current.shape)

            # normalize data
            for iImg in range(n_img):

                # normalize and baseline correction
                data_current_min = np.min(data_current[iImg, :])
                data_current_max = np.max(data_current[iImg, :])
                data_current[iImg, :] = (data_current[iImg, :] - data_current_min) / (data_current_max - data_current_min)

                # data_avg = np.min(abs(data_current[iImg, start-3]))
                # data_current[iImg, :] = abs(data_current[iImg, :]) - data_avg

                # axs[iC].plot(np.arange(data_current.shape[1])-start_plot, data_current[iImg, :], color=color_layer[iL], lw=0.2, alpha=0.2)
            
            # compute
            data_mean = np.mean(data_current[:, :], 0)
            data_sem = np.std(data_current[start_plot:, :], 0)/math.sqrt(n_img)

            # visualize
            axs[iC].plot(np.arange(data_current.shape[1])-start_plot, data_mean, color=color_layer[iL], markersize=1, marker='o')

            # plot stimulus
            axs[iC].tick_params(axis='both', labelsize=fontsize_tick)
            axs[iC].axhline(0, lw=0.5, color='grey')
            axs[iC].axvline(start-start_plot-1, lw=0.5, color='grey')   
            axs[iC].spines['top'].set_visible(False)
            axs[iC].spines['right'].set_visible(False)
            axs[iC].set_ylim(-0.1, 1.1)
            if (iC > 0):
                axs[iC].spines['left'].set_visible(False)
                axs[iC].spines['right'].set_visible(False)
                axs[iC].set_yticks([])
            if iL == 0:
                axs[iC].axvspan(0, tempCond[iC], facecolor='lightgrey', edgecolor=None, alpha=0.5)
            
    # save
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig2A_model', dpi=600)
    plt.savefig(root + 'visualization/Fig2A_model.svg')
    plt.close()

def plot_dynamics(dynamics_lin, dynamics_log, metric, tempCond, t1_plot, n_layer, color, n_img, init, computation, output_mode, root):

    # fontsizes 
    fontsize_title          = 20
    fontsize_legend         = 12
    fontsize_label          = 15
    fontsize_tick           = 10


    # join dynamics
    dynamics = [dynamics_lin, dynamics_log]

    # initiate figure
    fig, axs = plt.subplots(1, n_layer, figsize=(10, 3))
    sns.despine(offset=10)

    linestyle = ['dashed', 'dashdot']
    color_fit = ['lightgrey', 'darkgrey']
    lw = 2

    for iL in range(n_layer):

        # select data
        data_current = metric[:, iL, :, :].mean(2)

        # compute spread
        data_mean = np.mean(data_current, 0)
        data_sem = np.std(data_current, 0)/math.sqrt(init)
        
        # plot
        axs[iL].scatter(tempCond, data_mean, facecolor=color[iL], edgecolor='white', s=80, zorder=1)

        # plot spread
        for iC in range(len(tempCond)):
            axs[iL].plot([tempCond[iC], tempCond[iC]], [data_mean[iC] - data_sem[iC], data_mean[iC] + data_sem[iC]], color=color[iL], zorder=-10)

        # plot dynamics fit
        for iComp, comp in enumerate(computation):

            # select data
            data_current = dynamics[iComp][:, iL, :, :].mean(1)

            # compute
            data_mean = np.mean(data_current, 0)

            # plot
            axs[iL].plot(t1_plot, data_mean, color=color_fit[iComp], lw=lw, label=comp, linestyle=linestyle[iComp], zorder=-10)

        # adjust axes
        axs[iL].tick_params(axis='both', labelsize=fontsize_tick)  
        axs[iL].spines['top'].set_visible(False)
        axs[iL].spines['right'].set_visible(False)
        axs[iL].set_xticks(tempCond)
        axs[iL].set_xticklabels(tempCond, rotation=45)
        # axs[iL].set_title(output_mode + str(iL+1), fontsize=fontsize_title)
        # if iL == 1:
        #     axs[iL].set_xlabel('Duration (s)', fontsize=fontsize_label)
        if iL == 0:    
            axs[iL].set_ylabel('Response magnitude', fontsize=fontsize_label)

    # save fig
    axs[0].legend(fontsize=fontsize_legend, frameon=False)
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig2E_model', dpi=600)
    plt.savefig(root + 'visualization/Fig2E_model.svg')
    plt.close()

def plot_dynamics_metric(dynamics_fit, n_layer, computation, color, n_img, init, output_mode, root):

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
    offset_L        = [0, 4, 8, 12]

    patchs = list()
    for iL in range(n_layer):

        # statistical analysis
        sample1 = dynamics_fit[:, iL, :, 0].mean(0)
        sample2 = dynamics_fit[:, iL, :, 1].mean(0)
        result = stats.ttest_rel(sample1, sample2)
        print(result)

        for iC, comp in enumerate(computation):

            # select data
            data_current = dynamics_fit[:, iL, :, iC].mean(1)

            # compute
            data_mean = np.mean(data_current)
            data_sem = np.std(data_current)/math.sqrt(init)

            # plot
            patch = ax.bar(offset_L[iL]+offset_comp[iC], data_mean, color=color[iL], edgecolor='white', label=output_mode + str(iL+1), hatch=hatch[iC], alpha=alpha[iC])
            patchs.append(patch)

            # plot spread
            ax.plot([offset_L[iL]+offset_comp[iC], offset_L[iL]+offset_comp[iC]], [data_mean - data_sem, data_mean], color='black')

    # adjust axes
    ax.tick_params(axis='both', labelsize=fontsize_tick)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([offset_L[0]+0.5, offset_L[1]+0.5, offset_L[2]+0.5, offset_L[3]+0.5])
    ax.set_xticklabels(['E1', 'E2', 'E3', 'E4'])
    ax.set_ylabel(r'R$^{2}$', fontsize=fontsize_label)

    # save figure
    # ax.legend([(patchs[2], patchs[3], patchs[4]), (patchs[1], patchs[3], patchs[5])], [computation[0], computation[1]], frameon=False, fontsize=fontsize_legend, 
    #           bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    # ax.legend(frameon=False, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig2F_model', dpi=600)
    plt.savefig(root + 'visualization/Fig2F_model.svg')
    plt.close()

def plot_regression(ratio_lin_log, ratio_trans_sust, n_layer, color, n_img, init, model, root):

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

    # statistics
    print('\n')
    for i in range(2):
        if i == 0:
            print('Ratio lin/log')
            result = f_oneway(ratio_lin_log[:, 0, :].mean(0), ratio_lin_log[:, 1, :].mean(0), ratio_lin_log[:, 2, :].mean(0), ratio_lin_log[:, 3, :].mean(0))
            print(result)
            res = tukey_hsd(ratio_lin_log[:, 0, :].mean(0), ratio_lin_log[:, 1, :].mean(0), ratio_lin_log[:, 2, :].mean(0), ratio_lin_log[:, 3, :].mean(0))
            print(res)
        else:
            print('Ratio trans/sust')
            result = f_oneway(ratio_trans_sust[:, 0, :].mean(0), ratio_trans_sust[:, 1, :].mean(0), ratio_trans_sust[:, 2, :].mean(0), ratio_trans_sust[:, 3, :].mean(0))
            print(result)
            res = tukey_hsd(ratio_trans_sust[:, 0, :].mean(0), ratio_trans_sust[:, 1, :].mean(0), ratio_trans_sust[:, 2, :].mean(0), ratio_trans_sust[:, 3, :].mean(0))
            print(res)

    # set x and y limits
    lmts = [[0.5, 1.3], [-5, 60]]

    # visualize correlations
    data_per_layer = []
    for iL in range(n_layer):

        # select data
        data_current_x = np.mean(ratio_lin_log[:, iL, :], 0)
        data_current_y = np.mean(ratio_trans_sust[:, iL, :], 0)

        # visualize
        ax['regression'].scatter(data_current_x, data_current_y, facecolor=color[iL], linewidths=0.5, edgecolor='white', s=20) #, alpha=0.5)

        # concat
        data_current = [data_current_x, data_current_y]
        data_per_layer.append(data_current)

    # visualize averages
    for iL in range(n_layer):
        for i in range(2): 

            # visualize averages
            data_mean = np.mean(data_per_layer[iL][i])
            data_sem = np.std(data_per_layer[iL][i])/math.sqrt(init)

            if i == 0:
                ax_avg[i].plot([data_mean - data_sem, data_mean + data_sem], [iL, iL], color=color[iL], lw=0.75, zorder=-10)
            else:
                ax_avg[i].plot([iL, iL], [data_mean - data_sem, data_mean + data_sem], color=color[iL], lw=0.75, zorder=-10)

            # plot mean
            if i == 0:
                ax_avg[i].scatter(data_mean, iL, edgecolors=color[iL], facecolor='white', s=20, zorder=1)
            else:
                ax_avg[i].scatter(iL, data_mean, edgecolors=color[iL], facecolor='white', linewidth=1, s=20, zorder=1)

            # adjust axes
            ax_avg[i].set_xticks([])
            ax_avg[i].set_yticks([])      
            if i == 0:
                ax_avg[i].spines['top'].set_visible(False)
                ax_avg[i].spines['right'].set_visible(False)
                ax_avg[i].spines['left'].set_visible(False)
                ax_avg[i].spines['bottom'].set_color('grey')
                ax_avg[i].set_xlim(lmts[i][0], lmts[i][1])
                ax_avg[i].set_ylim(-1, 4)
            else:
                ax_avg[i].spines['top'].set_visible(False)
                ax_avg[i].spines['bottom'].set_visible(False)
                ax_avg[i].spines['right'].set_visible(False)
                ax_avg[i].spines['left'].set_color('grey')
                ax_avg[i].set_ylim(lmts[i][0], lmts[i][1])
                ax_avg[i].set_xlim(-1, 4)

    ax['regression'].set_xlim(lmts[0][0], lmts[0][1])
    ax['regression'].set_ylim(lmts[1][0], lmts[1][1])

    ################################################################### ALL

    # plot regression line
    x = ratio_lin_log[:, 0, :].mean(0).tolist() + ratio_lin_log[:, 1, :].mean(0).tolist() + ratio_lin_log[:, 2, :].mean(0).tolist() + ratio_lin_log[:, 3, :].mean(0).tolist()
    y = ratio_trans_sust[:, 0, :].mean(0).tolist() + ratio_trans_sust[:, 1, :].mean(0).tolist() + ratio_trans_sust[:, 2, :].mean(0).tolist() + ratio_trans_sust[:, 3, :].mean(0).tolist()

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
    ax['regression'].plot(np.linspace(np.min(x), np.max(x), 100), y, color='grey', lw=2, linestyle='dashed')

    # adjust axes
    ax['regression'].tick_params(axis='both', labelsize=fontsize_tick)
    ax['regression'].spines['top'].set_visible(False)
    ax['regression'].spines['right'].set_visible(False)
    ax['regression'].set_ylabel('Ratio transient/sustained', fontsize=fontsize_label)
    ax['regression'].set_xlabel('Ratio fit lin/log', fontsize=fontsize_label)

    # savefigur
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig2G_model', dpi=600)
    plt.savefig(root + 'visualization/Fig2G_model.svg')
    plt.close()

def plot_regression_CEandSC(ratio_lin_log, ratio_trans_sust, CEandSC_values, n_layer, color, n_img, model, root):

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
            if len(ratio_trans_sust.shape) == 3:
                data_current_y = ratio_trans_sust[:, iL, :].mean(0)
            else:
                data_current_y = ratio_trans_sust[iL, :]

            # visualize
            axs[row, 2*iL_col+iSumStat].scatter(data_current_x, data_current_y, edgecolor='white', facecolor='grey', s=100, linewidth=1.5)#, alpha=0.7)

            # select data
            x = data_current_x
            y = data_current_y
            
            # fit linear regression
            model = LinearRegression().fit(x.reshape(-1, 1), y)

            # slope and statistics
            slope = model.coef_[0]
            # print("Slope:", slope)

            x_with_const = sm.add_constant(x)
            model_sm = sm.OLS(y, x_with_const).fit()
            if iL == 0:
                print('Layer: ', iL+1, ', statistic: ', SumStat)
                print(model_sm.summary())

            # predict line
            y = model.intercept_ + model.coef_*np.linspace(np.min(x), np.max(x), 100)
            sc = axs[row, 2*iL_col+iSumStat].plot(np.linspace(np.min(x), np.max(x), 100), y, color='crimson', lw=4, linestyle='dashed')

            # adjust axes
            axs[row, 2*iL_col+iSumStat].tick_params(axis='both', labelsize=fontsize_tick)
            axs[row, 2*iL_col+iSumStat].spines['top'].set_visible(False)
            axs[row, 2*iL_col+iSumStat].spines['right'].set_visible(False)
            # axs[row, 2*iL_col+iSumStat].set_xticks([])
            if row == 1:
                axs[row, 2*iL_col+iSumStat].set_xlabel(SumStat, fontsize=fontsize_label)
            if (iSumStat == 0) & ((iL == 0) | (iL == 2)):
                axs[row, 2*iL_col+iSumStat].set_ylabel('Ratio transient/sustained', fontsize=fontsize_label)
            if iSumStat == 1:
                axs[row, 2*iL_col+iSumStat].set_yticks([])

    # save figure
    fig.align_ylabels()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    plt.savefig(root + 'visualization/SFig2_model', dpi=600)
    plt.savefig(root + 'visualization/SFig2_model.svg')
    plt.close()

def plot_regression_CEandSC_L1(ratio_lin_log, ratio_trans_sust, CEandSC_values, n_layer, color, n_img, model, root, Groen2013=False):

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
    for iSumStat, SumStat in enumerate(summary_statistics):

        print(SumStat)

        # select data
        data_current_x = CEandSC_values[:, iSumStat]
        if len(ratio_trans_sust.shape) == 3:
            data_current_y = ratio_trans_sust[:, 0, :].mean(0)
        else:
            data_current_y = ratio_trans_sust[0, :]

        # visualize
        axs[iSumStat].scatter(data_current_x, data_current_y, color='white', facecolor='grey', s=100, linewidth=1.5)#, alpha=0.7)

        # select data
        x = data_current_x
        y = data_current_y
        
        # fit linear regression
        model = LinearRegression().fit(x.reshape(-1, 1), y)

        # slope and statistics
        slope = model.coef_[0]
        # print("Slope:", slope)

        x_with_const = sm.add_constant(x)
        model_sm = sm.OLS(y, x_with_const).fit()
        print(model_sm.summary())

        # predict line
        y = model.intercept_ + model.coef_*np.linspace(np.min(x), np.max(x), 100)
        sc = axs[iSumStat].plot(np.linspace(np.min(x), np.max(x), 100), y, color='crimson', lw=3, linestyle='dashed')

        # adjust axes
        axs[iSumStat].tick_params(axis='both', labelsize=fontsize_tick)
        axs[iSumStat].spines['top'].set_visible(False)
        axs[iSumStat].spines['right'].set_visible(False)
        axs[iSumStat].set_xlabel(SumStat, fontsize=fontsize_label)

    # save figure
    fig.align_ylabels()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    if Groen2013:
        plt.savefig(root + 'visualization/Fig4D', dpi=600)
        plt.savefig(root + 'visualization/Fig4D.svg')
    else:
        plt.savefig(root + 'visualization/Fig4B', dpi=600)
        plt.savefig(root + 'visualization/Fig4B.svg')
    plt.close()

def stats_lin_log(n_layer, dynamics_fit):

    alpha = 0.05

    for iL in range(n_layer):

        # print progress
        print('Layer', iL+1)

        # ttest
        sample1 = dynamics_fit[:, iL, :, 0].mean(0)
        sample2 = dynamics_fit[:, iL, :, 1].mean(0)
        p = stats.ttest_rel(sample1, sample2)
        if p[1] < alpha:
            print('Linear vs. log fit', p, ' SIGNIFICANT')
        else:
            print('Linear vs. log fit', p)

        print('\n')
