import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from neural_data_visualize_utils import *
from __model_visualize_onepulse_utils import *

# set root
root        = '/home/amber/OneDrive/code/prednet_Brands/' 

# stimulus dataset
dataset_stim = ['set1', 'set2']

# select model
model = 'Lotter2017'
# model = 'Kirubeswaran2023'

# training set
training_set = 'random'
training_set = 'KITTI'
training_set = 'WT_AMS'
training_set = 'WT_VEN'
training_set = 'WT_WL'

# set directory's
if model == 'Lotter2017':
    root_data       = '/home/amber/Documents/prednet_Brands2024/data/model/Lotter2017/'
    root_vis        = '/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Lotter2017/onepulse/'
    init                = 1
elif model == 'Kirubeswaran2023':
    root_data       = '/home/amber/Documents/prednet_Brands2024/data/model/Kirubeswaran2023/datasets/' + training_set + '/'
    root_vis        = '/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Kirubeswaran2023/datasets/' + training_set + '/'
    init                = 5

# set trained or untrained
trained = True

# experimental info
nt                  = 45
n_img               = 48
start               = 3
tempCond            = np.array([1, 2, 4, 8, 16, 32], dtype=int)

n_layer             = 4

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
for i, datasetstim in enumerate(dataset_stim):
    if i == 0:
        CEandSC_values[:int(n_img/2)] = np.load('/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/' + datasetstim + '.npy')
    else:
        CEandSC_values[int(n_img/2):] = np.load('/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/' + datasetstim + '.npy')

# plot settings
# color_tempCond      = ['#9BD2E1', '#81C4E7', '#7EB2E4', '#9398D2', '#9D7DB2', '#906388']
# color_layer         = ['#A6BE54', '#D1B541', '#E49C39', '#E67932']

color_tempCond      = ['#9BD2E1', '#81C4E7', '#7EB2E4', '#9398D2', '#9D7DB2', '#906388']
color_layer         = ['#69B190', '#549EB3', '#4E79C5', '#6F4C9B']

# computations
computation                = ['linear fit', 'log fit']

# fit curve for recovery of adaptation initial parameter values
t1_plot     = np.linspace(min(tempCond), max(tempCond), 1000)
p0          = [1, 0]

# initiate dataframes
data                = np.zeros((init, n_layer, len(tempCond), n_img, nt))
metric              = np.zeros((init, n_layer, len(tempCond), n_img))

dynamics_lin        = np.zeros((init, n_layer, n_img, len(t1_plot)))
dynamics_log        = np.zeros((init, n_layer, n_img, len(t1_plot)))
dynamics_fit        = np.zeros((init, n_layer, n_img, len(computation)))

ratio_lin_log       = np.zeros((init, n_layer, n_img))
ratio_trans_sust    = np.zeros((init, n_layer, n_img))

# time window ranges (longest duration)
range_trans     = [start, start+2]
range_sust      = [start+tempCond[-1]-20, start+tempCond[-1]]

# plot model activations
for iInit in range(init):

    lbls = []
    for iL in range(n_layer):

        # import trials onepulse
        for i, datasetstim in enumerate(dataset_stim):

            # retrieve data
            if model == 'Lotter2017':
                if population == None:
                    if trained:
                        temp = np.load(root_data + datasetstim + '/avgs/prednet_onepulse_' + output_mode + str(iL+1) + '_actvs_KITTI.npy')
                    else:
                        temp = np.load(root_data + datasetstim + '/avgs/prednet_onepulse_' + output_mode + str(iL+1) + '_actvs_random_' + str(iInit + 1) + '.npy')
                else:
                    if trained:
                        temp = np.load(root_data + datasetstim + '/avgs/prednet_onepulse_' + output_mode + str(iL+1) + population + '_actvs_KITTI.npy')
                    else:
                        temp = np.load(root_data + datasetstim + '/avgs/prednet_onepulse_' + output_mode + str(iL+1) + population + '_actvs_' + str(iInit + 1) + '_random.npy')
            else:
                if population == None:
                    if trained:
                        temp = np.load(root_data + 'static_' + datasetstim + '_onepulse_' + output_mode + str(iL+1) + '_actvs_trained_' + str(iInit + 1) + '.npy')
                    else:
                        temp = np.load(root_data + datasetstim + '_onepulse_' + output_mode + str(iL+1) + '_actvs_random_' + str(iInit + 1) + '.npy')
                else:
                    if trained:
                        temp = np.load(root_data + datasetstim + '_onepulse_' + output_mode + str(iL+1) + population + '_actvs_' + str(iInit + 1) + '_trained.npy')
                    else:
                        temp = np.load(root_data + datasetstim + '_onepulse_' + output_mode + str(iL+1) + population + '_actvs_' + str(iInit + 1) + '_random.npy')
                
            # store data
            if i == 0:
                data[iInit, iL, :, :int(n_img/2), :] = temp
            else:
                data[iInit, iL, :, int(n_img/2):, :] = temp

        # compute TRANS:SUST ratio
        for iImg in range(n_img):

            # compute T:S ratio
            data_current = data[iInit, iL, -1, iImg, :]

            # compute peak and transient
            trans = np.max(data_current[range_trans[0]:range_trans[1]])
            sust = np.mean(data_current[range_sust[0]:range_sust[1]])
            ratio_trans_sust[iInit, iL, iImg] = trans/sust

        # compute FIT
        for iC in range(len(tempCond)):

            # select data
            data_current = data[iInit, iL, iC, :, :]

            # compute metric over ALL stim. dur.
            for iImg in range(n_img):
                    
                # compute metric
                data_baseline_correct = data_current[iImg, start:start+tempCond[iC]]
                metric[iInit, iL, iC, iImg] = np.sum(data_baseline_correct)/np.sum(data[iInit, iL, -1, start:start+tempCond[-1]])

            # compute line fit
            for iImg in range(n_img):

                # fit curve - lin
                popt, _ = curve_fit(OF_dynamics_linear, tempCond, metric[iInit, iL, :, iImg], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                dynamics_lin[iInit, iL, iImg, :] = OF_dynamics_linear(t1_plot, *popt)

                pred = OF_dynamics_linear(tempCond, *popt)
                dynamics_fit[iInit, iL, iImg, 0] = r_squared(metric[iInit, iL, :, iImg], pred)

                # fit curve - log
                popt, _ = curve_fit(OF_dynamics_log, tempCond, metric[iInit, iL, :, iImg], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                dynamics_log[iInit, iL, iImg, :] = OF_dynamics_log(t1_plot, *popt)

                pred = OF_dynamics_log(tempCond, *popt)
                dynamics_fit[iInit, iL, iImg, 1] = r_squared(metric[iInit, iL, :, iImg], pred)

                # copmute ratio
                ratio_lin_log[iInit, iL, iImg] = dynamics_fit[iInit, iL, iImg, 0]/dynamics_fit[iInit, iL, iImg, 1]

######################## VISUALIZE
# plot_broadband(data, start, tempCond, n_layer, n_img, range_trans, range_sust, trained, population, None, output_mode, color_tempCond, color_layer, root_data, root_vis)

# plot_dynamics(dynamics_lin, dynamics_log, metric, tempCond, t1_plot, n_layer, color_layer, n_img, init, computation, output_mode, root_vis)
# plot_dynamics_metric(dynamics_fit, n_layer, computation, color_layer, n_img, init, output_mode, root_vis)
plot_regression(ratio_lin_log, ratio_trans_sust, n_layer, color_layer, n_img, init, model, root_vis)

# plot_regression_CEandSC(ratio_lin_log, ratio_trans_sust, CEandSC_values, n_layer, color_layer, n_img, model, root_vis)
# plot_regression_CEandSC_L1(ratio_lin_log, ratio_trans_sust, CEandSC_values, n_layer, color_layer, n_img, model, root_vis)


# ######################## STATISTICS
# stats_lin_log(n_layer, dynamics_fit)
# stats_regression(ratio_lin_log, ratio_trans_sust)

######################## SAVE
# np.save(root_data_save + 'ratio_lin_log_' + model + '_' + dataset, ratio_lin_log) 
# np.save(root_data_save + 'ratio_trans_sust_' + model + '_' + dataset, ratio_trans_sust)