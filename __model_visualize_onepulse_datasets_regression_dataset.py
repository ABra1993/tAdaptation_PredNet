import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from neural_data_visualize_utils import *
from __model_visualize_onepulse_datasets_utils import *


# training set
training_sets = ['random', 'KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']

# set root
root                = '/home/amber/OneDrive/code/prednet_Brands/' 

root_data           = '/home/amber/Documents/prednet_Brands2024/data/model/Kirubeswaran2023/datasets/analyse_dataset/'
root_vis            = '/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Kirubeswaran2023/datasets/analyse_dataset/'

# select model
model = 'Kirubeswaran2023'

# test set
test_sets = 'sets'
# test_sets = 'datasets'

if test_sets == 'sets':
    test_sets_stim          = ['set1', 'set2']

elif test_sets == 'datasets':
    test_sets_stim          = ['dataset']

# temporal input
input_motion        = ['dynamic', 'static']

# set trained or untrained
preload = True

# experimental info
nt                  = 45
start               = 3
tempCond            = np.array([1, 2, 4, 8, 16, 32], dtype=int)
n_img               = 48
n_layer             = 4
init            = 5

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
color_dataset       = ['#F6C141', '#4EB265', '#5289C7', '#DC050C']

# computations
computation                = ['linear fit', 'log fit']

# fit curve for recovery of adaptation initial parameter values
t1_plot     = np.linspace(min(tempCond), max(tempCond), 1000)
p0          = [1, 0]

# import
if preload == False:

    # initiate dataframes
    data                = np.zeros((len(training_sets), len(input_motion), init, n_layer, len(tempCond), n_img, nt))
    metric              = np.zeros((len(training_sets), len(input_motion), init, n_layer, len(tempCond), n_img))

    dynamics_lin        = np.zeros((len(training_sets), len(input_motion), init, n_layer, n_img, len(t1_plot)))
    dynamics_log        = np.zeros((len(training_sets), len(input_motion), init, n_layer, n_img, len(t1_plot)))
    dynamics_fit        = np.zeros((len(training_sets), len(input_motion), init, n_layer, n_img, len(computation)))

    ratio_lin_log       = np.zeros((len(training_sets), len(input_motion), init, n_layer, n_img))
    ratio_trans_sust    = np.zeros((len(training_sets), len(input_motion), init, n_layer, n_img))

    # time window ranges (longest duration)
    range_trans     = [start, start+2]
    range_sust      = [start+tempCond[-1]-10, start+tempCond[-1]]

    # plot model activations
    for iTS, training_set in enumerate(training_sets):

        if training_set == 'random':
            continue

        # root dataset
        if 'KITTI' in training_set:
            dataset = 'KITTI'
        elif 'WT_AMS' in training_set:
            dataset = 'WT_AMS'
        elif 'WT_VEN' in training_set:
            dataset = 'WT_VEN'
        elif 'WT_WL' in training_set:
            dataset = 'WT_WL'

        for iM, motion in enumerate(input_motion):
            for iInit in range(init):

                lbls = []
                for iL in range(n_layer):

                    # import trials onepulse
                    for i, datasetstim in enumerate(test_sets_stim):

                        # retrieve data
                        if motion == 'static':
                            temp = np.load(root_data + 'activations/' + dataset + '/static_' + datasetstim + '_onepulse_' + output_mode + str(iL+1) + '_actvs_trained_' + str(iInit + 1) + '.npy')
                        
                        elif motion == 'dynamic':
                            temp = np.load(root_data + 'activations/' + dataset + '/' + datasetstim + '_onepulse_' + output_mode + str(iL+1) + '_actvs_trained_' + str(iInit + 1) + '.npy')
                        
                        # store data
                        if test_sets == 'sets':
                            if i == 0:
                                data[iTS, iM, iInit, iL, :, :int(n_img/2), :] = temp
                            else:
                                data[iTS, iM, iInit, iL, :, int(n_img/2):, :] = temp
                        elif test_sets == 'datasets':
                            data[iTS, iM, iInit, iL, :, :, :] = temp

                    # compute TRANS:SUST ratio
                    for iImg in range(n_img):

                        # compute T:S ratio
                        data_current = data[iTS, iM, iInit, iL, -1, iImg, :]

                        # compute peak and transient
                        trans = np.max(data_current[range_trans[0]:range_trans[1]])
                        sust = np.mean(data_current[range_sust[0]:range_sust[1]])
                        ratio_trans_sust[iTS, iM, iInit, iL, iImg] = trans/sust
                        # ratio_trans_sust[iTS, iM, iInit, iL, iImg] = np.sum(data_current)**2

                    # compute FIT
                    for iC in range(len(tempCond)):

                        # select data
                        data_current = data[iTS, iM, iInit, iL, iC, :, :]

                        # compute metric over ALL stim. dur.
                        for iImg in range(n_img):
                                
                            # compute metric
                            data_baseline_correct = data_current[iImg, start:start+tempCond[iC]]
                            metric[iTS, iM, iInit, iL, iC, iImg] = np.sum(data_baseline_correct)/np.sum(data[iTS, iM, iInit, iL, -1, start:start+tempCond[-1]])

                        # compute line fit
                        for iImg in range(n_img):

                            # fit curve - lin
                            popt, _ = curve_fit(OF_dynamics_linear, tempCond, metric[iTS, iM, iInit, iL, :, iImg], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                            dynamics_lin[iTS, iM, iInit, iL, iImg, :] = OF_dynamics_linear(t1_plot, *popt)

                            pred = OF_dynamics_linear(tempCond, *popt)
                            dynamics_fit[iTS, iM, iInit, iL, iImg, 0] = r_squared(metric[iTS, iM, iInit, iL, :, iImg], pred)

                            # fit curve - log
                            popt, _ = curve_fit(OF_dynamics_log, tempCond, metric[iTS, iM, iInit, iL, :, iImg], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                            dynamics_log[iTS, iM, iInit, iL, iImg, :] = OF_dynamics_log(t1_plot, *popt)

                            pred = OF_dynamics_log(tempCond, *popt)
                            dynamics_fit[iTS, iM, iInit, iL, iImg, 1] = r_squared(metric[iTS, iM, iInit, iL, :, iImg], pred)

                            # copmute ratio
                            ratio_lin_log[iTS, iM, iInit, iL, iImg] = dynamics_fit[iTS, iM, iInit, iL, iImg, 1]/dynamics_fit[iTS, iM, iInit, iL, iImg, 0]


    ####################### SAVE
    np.save(root_data + '/metrics/onepulse_ratio_lin_log', ratio_lin_log) 
    np.save(root_data + '/metrics/onepulse_ratio_trans_sust', ratio_trans_sust)
    print(ratio_lin_log.shape)

# load data
ratio_lin_log               = np.load(root_data + '/metrics/onepulse_ratio_lin_log.npy') 
ratio_trans_sust            = np.load(root_data + '/metrics/onepulse_ratio_trans_sust.npy')
# print(ratio_lin_log.shape)

######################## VISUALIZE
plot_regression_dataset_all(test_sets, ratio_lin_log[:, 0, :, :, :], ratio_trans_sust[:, 0, :, :, :], training_sets, input_motion, n_layer, color_layer, n_img, init, root_vis)


# plot_regression_dataset_staticmotion(test_sets, ratio_lin_log, ratio_trans_sust, training_sets, input_motion, n_layer, color_layer, n_img, init, root_vis)
# plot_regression_dataset(test_sets, ratio_lin_log[:, 0, :, :, :], ratio_trans_sust[:, 0, :, :, :], training_sets, input_motion, n_layer, color_layer, n_img, init, root_vis)
# plot_regression_CEandSC_L1_all(test_sets, ratio_lin_log[:, 0, :, :, :], ratio_trans_sust[:, 0, :, :, :], training_sets, input_motion, CEandSC_values, n_layer, color_dataset, n_img, init, root_vis)


# ----------------------------------------------------- OLD
# plot_broadband(data, start, tempCond, n_layer, n_img, range_trans, range_sust, trained, population, None, output_mode, color_tempCond, color_layer, root_data, root_vis)
# plot_dynamics(dynamics_lin, dynamics_log, metric, tempCond, t1_plot, n_layer, color_layer, n_img, init, computation, output_mode, root_vis)
# plot_dynamics_metric(dynamics_fit, n_layer, computation, color_layer, n_img, init, output_mode, root_vis)
# plot_regression(ratio_lin_log, ratio_trans_sust, n_layer, color_layer, n_img, init, model, root_vis)
# plot_regression_CEandSC(ratio_lin_log, ratio_trans_sust, CEandSC_values, n_layer, color_layer, n_img, model, root_vis)



# ######################## STATISTICS
# # stats_lin_log(n_layer, dynamics_fit)
# stats_regression(ratio_lin_log, ratio_trans_sust)

