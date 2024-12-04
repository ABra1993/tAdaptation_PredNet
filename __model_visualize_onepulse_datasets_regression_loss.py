import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from neural_data_visualize_utils import *
from __model_visualize_onepulse_datasets_utils import *

# dataset
training_sets = ['KITTI_Lnull', 'KITTI_Lall']
training_sets = ['WT_AMS_Lnull', 'WT_AMS_Lall']
training_sets = ['WT_VEN_Lnull', 'WT_VEN_Lall']
training_sets = ['WT_WL_Lnull', 'WT_WL_Lall']

# set root
root                = '/prednet_Brands2024_git/' 

root_data           = '/prednet_Brands2024_git/data/model/Kirubeswaran2023/datasets/analyse_loss/'
root_vis            = '/prednet_Brands2024_git/visualization/model/Kirubeswaran2023/datasets/analyse_loss/'

# select model
model = 'Kirubeswaran2023'

# root for save
if 'KITTI' in training_sets[0]:
    dataset = 'KITTI'
elif 'WT_AMS' in training_sets[0]:
    dataset = 'WT_AMS'
elif 'WT_VEN' in training_sets[0]:
    dataset = 'WT_VEN'
elif 'WT_WL' in training_sets[0]:
    dataset = 'WT_WL'

# test set
test_sets = 'sets'
# test_sets = 'datasets'

if test_sets == 'sets':
    test_sets_stim          = ['set1', 'set2']

elif test_sets == 'datasets':
    test_sets_stim          = ['dataset']

# set trained or untrained
preload = False

# experimental info
nt                  = 45
start               = 3
tempCond            = np.array([1, 2, 4, 8, 16, 32], dtype=int)
n_img               = 48
n_layer             = 4
init                = 3

# output mode
output_mode         = 'E'
# output_mode         = 'error'
# output_mode         = 'R'
# output_mode         = 'A'
# output_mode         = 'Ahat'

assert output_mode in ['E', 'R', 'A', 'Ahat', 'error']

# error population
# population = '+'
# population = '-'
population = None

# import CE and SC values
CEandSC_values = np.zeros((n_img, 2))
if test_sets == 'sets':
    for i, datasetstim in enumerate(test_sets_stim):
        if i == 0:
            CEandSC_values[:int(n_img/2)] = np.load('/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/' + datasetstim + '.npy')
        else:
            CEandSC_values[int(n_img/2):] = np.load('/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/' + datasetstim + '.npy')

# plot settings
# color_tempCond      = ['#9BD2E1', '#81C4E7', '#7EB2E4', '#9398D2', '#9D7DB2', '#906388']
# color_layer         = ['#A6BE54', '#D1B541', '#E49C39', '#E67932']

color_tempCond      = ['#9BD2E1', '#81C4E7', '#7EB2E4', '#9398D2', '#9D7DB2', '#906388']
color_layer         = ['#69B190', '#549EB3', '#4E79C5', '#6F4C9B']
color_dataset       = ['#DDCC77', '#117733', '#88CCEE', '#882255']

# computations
computation                = ['linear fit', 'log fit']

# fit curve for recovery of adaptation initial parameter values
t1_plot     = np.linspace(min(tempCond), max(tempCond), 1000)
p0          = [1, 0]

# import
if preload == False:

    # initiate dataframes
    data                = np.zeros((len(training_sets), init, n_layer, len(tempCond), n_img, nt))
    metric              = np.zeros((len(training_sets), init, n_layer, len(tempCond), n_img))

    dynamics_lin        = np.zeros((len(training_sets), init, n_layer, n_img, len(t1_plot)))
    dynamics_log        = np.zeros((len(training_sets), init, n_layer, n_img, len(t1_plot)))
    dynamics_fit        = np.zeros((len(training_sets), init, n_layer, n_img, len(computation)))

    ratio_lin_log       = np.zeros((len(training_sets), init, n_layer, n_img))
    ratio_trans_sust    = np.zeros((len(training_sets), init, n_layer, n_img))

    # time window ranges (longest duration)
    range_trans     = [start, start+2]
    range_sust      = [start+tempCond[-1]-10, start+tempCond[-1]]

    # plot model activations
    for iTS, training_set in enumerate(training_sets):

        if training_set == 'random':
            continue

        for iInit in range(init):

            lbls = []
            for iL in range(n_layer):

                # import trials onepulse
                for i, datasetstim in enumerate(test_sets_stim):

                    # retrieve data
                    temp = np.load(root_data + 'activations/' + dataset + '/' + datasetstim + '_' + training_set + '_onepulse_' + output_mode + str(iL+1) + '_actvs_trained_' + str(iInit + 1) + '.npy')

                    # store data
                    if test_sets == 'sets':
                        if i == 0:
                            data[iTS, iInit, iL, :, :int(n_img/2), :] = temp[:, :int(n_img/2), :]
                        else:
                            data[iTS, iInit, iL, :, int(n_img/2):, :] = temp[:, :int(n_img/2), :]
                    elif test_sets == 'datasets':
                        data[iTS, iInit, iL, :, :, :] = temp

                # compute TRANS:SUST ratio
                for iImg in range(n_img):

                    # compute T:S ratio
                    data_current = data[iTS, iInit, iL, -1, iImg, :]

                    # compute peak and transient
                    trans = np.max(data_current[range_trans[0]:range_trans[1]])
                    sust = np.mean(data_current[range_sust[0]:range_sust[1]])
                    ratio_trans_sust[iTS, iInit, iL, iImg] = trans/sust
                    # ratio_trans_sust[iTS, iM, iInit, iL, iImg] = np.sum(data_current)**2

                # compute FIT
                for iC in range(len(tempCond)):

                    # select data
                    data_current = data[iTS, iInit, iL, iC, :, :]

                    # compute metric over ALL stim. dur.
                    for iImg in range(n_img):
                            
                        # compute metric
                        data_baseline_correct = data_current[iImg, start:start+tempCond[iC]]
                        metric[iTS, iInit, iL, iC, iImg] = np.sum(data_baseline_correct)/np.sum(data[iTS, iInit, iL, -1, start:start+tempCond[-1]])

                    # compute line fit
                    for iImg in range(n_img):

                        # fit curve - lin
                        popt, _ = curve_fit(OF_dynamics_linear, tempCond, metric[iTS, iInit, iL, :, iImg], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                        dynamics_lin[iTS, iInit, iL, iImg, :] = OF_dynamics_linear(t1_plot, *popt)

                        pred = OF_dynamics_linear(tempCond, *popt)
                        dynamics_fit[iTS, iInit, iL, iImg, 0] = r_squared(metric[iTS, iInit, iL, :, iImg], pred)

                        # fit curve - log
                        popt, _ = curve_fit(OF_dynamics_log, tempCond, metric[iTS, iInit, iL, :, iImg], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                        dynamics_log[iTS, iInit, iL, iImg, :] = OF_dynamics_log(t1_plot, *popt)

                        pred = OF_dynamics_log(tempCond, *popt)
                        dynamics_fit[iTS, iInit, iL, iImg, 1] = r_squared(metric[iTS, iInit, iL, :, iImg], pred)

                        # copmute ratio
                        ratio_lin_log[iTS, iInit, iL, iImg] = dynamics_fit[iTS, iInit, iL, iImg, 1]/dynamics_fit[iTS, iInit, iL, iImg, 0]


    ####################### SAVE
    np.save(root_data + 'metrics/onepulse_ratio_lin_log_' + dataset, ratio_lin_log) 
    np.save(root_data + 'metrics/onepulse_ratio_trans_sust_' + dataset, ratio_trans_sust)

# # load data
# ratio_lin_log = np.load(root_data + 'metrics/onepulse_ratio_lin_log_' + dataset + '.npy') 
# ratio_trans_sust = np.load(root_data + 'metrics/onepulse_ratio_trans_sust_' + dataset + '.npy')

######################## VISUALIZE
# plot_regression_all_loss(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color_layer, n_img, init, root_vis, dataset)

# plot_regression_all_loss_old(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, n_layer, color_layer, n_img, init, root_vis, dataset)
# plot_regression_CEandSC_L1_all(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, None, CEandSC_values, n_layer, color_dataset, n_img, init, root_vis)
# plot_regression_CEandSC_L1_all(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, input_motion, CEandSC_values, n_layer, color_dataset, n_img, init, root_vis)
# plot_broadband(data, start, tempCond, n_layer, n_img, range_trans, range_sust, trained, population, None, output_mode, color_tempCond, color_layer, root_data, root_vis)
# plot_dynamics(dynamics_lin, dynamics_log, metric, tempCond, t1_plot, n_layer, color_layer, n_img, init, computation, output_mode, root_vis)
# plot_dynamics_metric(dynamics_fit, n_layer, computation, color_layer, n_img, init, output_mode, root_vis)
# plot_regression(ratio_lin_log, ratio_trans_sust, n_layer, color_layer, n_img, init, model, root_vis)
# plot_regression_CEandSC(ratio_lin_log, ratio_trans_sust, CEandSC_values, n_layer, color_layer, n_img, model, root_vis)

# ######################## STATISTICS
# # stats_lin_log(n_layer, dynamics_fit)
# stats_regression(ratio_lin_log, ratio_trans_sust)


### PLOT ALL DATASETS
datasets = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']
ratio_lin_log_all = np.zeros((len(datasets), len(training_sets), init, n_layer, n_img))

for iD, dataset in enumerate(datasets):
    ratio_lin_log_all[iD, :, :, :, :] = np.load(root_data + 'metrics/onepulse_ratio_lin_log_' + dataset + '.npy') 

######################## VISUALIZE
plot_regression_all_datasets_loss(test_sets, ratio_lin_log_all, None, training_sets, n_layer, color_layer, n_img, init, root_vis, datasets)

