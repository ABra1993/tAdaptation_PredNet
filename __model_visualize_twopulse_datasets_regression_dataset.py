import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from neural_data_visualize_utils import *
from __model_visualize_twopulse_datasets_utils import *

# training set
training_sets = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']

# set root
root                = '/prednet_Brands2024_git/' 

root_data           = '/prednet_Brands2024_git/data/model/Kirubeswaran2023/datasets/analyse_dataset/'
root_vis            = '/prednet_Brands2024_git/visualization/model/Kirubeswaran2023/datasets/analyse_dataset/'

# select model
model = 'Kirubeswaran2023'

# test set
test_sets = 'sets'

if test_sets == 'sets':
    test_sets_stim          = ['set1', 'set2']

# set trained or untrained
preload = False

# experimental info
nt                  = 45
n_img               = 48

init                = 3
n_layer             = 4

start               = 4
duration            = 1

trained             = True

# temporal conditions
trials              = np.array(['twopulse_repeat', 'twopulse_nonrepeat_same'])
tempCond            = np.array([1, 2, 4, 8, 16, 32], dtype=int)

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
color_trial         = ['dodgerblue', np.array([212, 170, 0])/255]

# set time window
time_window = 5

# # computations
computation                = ['log fit']

# fit curve for recovery of adaptation initial parameter values
t1_plot     = np.linspace(min(tempCond), max(tempCond), 1000)
p0          = [1, 0]

# import
if preload == False:
    # initiate dataframes
    data                = np.zeros((len(training_sets), len(trials), n_layer, len(tempCond), n_img, nt))

    metric              = np.zeros((len(training_sets), len(trials), n_layer, len(tempCond), n_img))

    dynamics_log        = np.zeros((len(training_sets), len(trials), n_layer, n_img, len(t1_plot)))
    dynamics_fit        = np.zeros((len(training_sets), len(trials), n_layer, n_img))

    adaptation_avg      = np.zeros((len(training_sets), len(trials), n_layer, n_img))

    # plot model activations
    for iTS, training_set in enumerate(training_sets):
        for iInit in range(init):

            lbls = []
            for iL in range(n_layer):
                for iT, trial in enumerate(trials):

                    # import trials onepulse
                    for i, datasetstim in enumerate(test_sets_stim):

                        # retrieve data
                        temp = np.load(root_data + 'activations/' + training_set + '/' + datasetstim + '_' + training_set + '_' + trial + '_' + output_mode + str(iL+1) + '_actvs_trained_' + str(iInit + 1) + '.npy')
                        
                        # store data
                        if i == 0:
                            data[iTS, iT, iL, :, :int(n_img/2), :] = temp
                        else:
                            data[iTS, iT, iL, :, int(n_img/2):, :] = temp

                    # obtain DATA
                    for iC, iC_val in enumerate(tempCond):

                        iC_select = 0

                        # compute RS
                        for iImg in range(n_img):

                            # normalize data
                            if iC > 2:
                                data_ISI6 = data[iTS, iT, iL, iC, iImg, :] #- np.mean(data[iTS, iT, iL, iC, iImg, start-1])
                            else:
                                data_ISI6 = data[iTS, iT, iL, -1, iImg, :] #- np.mean(data[iTS, iT, iL, -1, iImg, start-1])

                            data_ISI = data[iTS, iT, iL, iC, iImg, :] #- np.mean(data[iTS, iT, iL, iC, iImg, start-1])

                            # compute first and second pulse
                            first_pulse = np.zeros(nt)
                            first_pulse[:start + duration + time_window] = data_ISI6[:start + duration + time_window]
                            second_pulse = data_ISI - first_pulse

                            # compute max
                            AUC1 = np.max(first_pulse[start:start+duration+time_window])
                            AUC2 = np.max(second_pulse[start+duration+tempCond[iC]:start+tempCond[iC]+duration+time_window])

                            # compute metric
                            metric[iTS, iT, iL, iC, iImg] = AUC2/AUC1

                            if (iTS == 3) & (iL == 0) & (iImg == 22) & (iInit == 0) & (iC == 0):
                                
                                print(start+tempCond[iC]+duration)

                                # plot
                                plt.plot(data_ISI, color='black', label='data')
                                plt.plot(first_pulse, color='grey', label='first_pulse')
                                plt.plot(second_pulse, color='red', label='second pulse')


                                plt.axvline(start, linestyle='dashed', color='grey')
                                plt.axvspan(start, start+duration+time_window, color='silver', alpha=0.5)
                                plt.axvspan(start+tempCond[iC]+duration, start+tempCond[iC]+duration+time_window, color='peachpuff', alpha=0.5)

                                # print metric
                                print(AUC1)
                                print(AUC2)
                                print(metric[iTS, iT, iL, iC, iImg])

                                # savefig
                                plt.legend()
                                plt.savefig(root+'visualization/model/compute_recovery')

                        # compute line fit
                        for iImg in range(n_img):

                            # fit curve - log
                            popt, _ = curve_fit(OF_dynamics_log, tempCond, metric[iTS, iT, iL, :, iImg], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                            dynamics_log[iTS, iT, iL, iImg, :] = OF_dynamics_log(t1_plot, *popt)

                            pred = OF_dynamics_log(tempCond, *popt)
                            dynamics_fit[iTS, iT, iL, iImg] = r_squared(metric[iTS, iT, iL, :, iImg], pred)

                    # compute average
                    adaptation_avg[iTS, iT, iL, :] = np.mean(metric[iTS, iT, iL, :, :], 0)

    ####################### SAVE
    np.save(root_data + 'metrics/twopulse_data', data) 
    np.save(root_data + 'metrics/twopulse_dynamics_log', dynamics_log) 
    np.save(root_data + 'metrics/twopulse_metric', metric) 
    np.save(root_data + 'metrics/twopulse_adaptation_avg', adaptation_avg) 


# load
data = np.load(root_data + 'metrics/twopulse_data.npy') 
dynamics_log = np.load(root_data + 'metrics/twopulse_dynamics_log.npy')
metric = np.load(root_data + 'metrics/twopulse_metric.npy')
adaptation_avg = np.load(root_data + 'metrics/twopulse_adaptation_avg.npy')

# ######################## VISUALIZE
# # plot regression
# plot_broadband_all(data, trials_lbls, start, tempCond, duration, n_layer, n_img, population, dataset, output_mode, color_trial, root_data, root_vis)

idx = 1
# plot_broadband(data[idx, :], training_sets[idx], trials, start, tempCond, duration, n_img, color_trial, root_vis)

# plot_dynamics(dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root_vis)
plot_dynamics_metric_all_dataset(training_sets, 'dataset', dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root_vis)
# plot_dynamics_all_dataset(training_sets, 'dataset', dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root_vis)


# # plot_regression_CEandSC(ratio_lin_log, metric, CEandSC_values, n_layer, color_layer, n_img, model, root_vis)

# ######################## STATISTICS
stats_trial(training_sets, adaptation_avg, n_layer)
