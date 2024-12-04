import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from neural_data_visualize_utils import *
from __model_visualize_twopulse_utils import *

# set root
root        = '/home/amber/OneDrive/code/prednet_Brands2024/' 

# stimulus dataset
dataset_stim = ['set1', 'set2']

# select model
model = 'Lotter2017'

# set directory's
if model == 'Lotter2017':
    root_data       = '/home/amber/Documents/prednet_Brands2024/data/model/Lotter2017/'
    root_vis        = '/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Lotter2017/twopulse/'

# set training data
if model == 'Lotter2017':
    dataset = 'KITTI'

trials = np.array(['twopulse_repeat', 'twopulse_nonrepeat_same'])
trials_lbls = ['repetition', 'alternation (within)']

# set trained or untrained
trained = True

# experimental info
nt                  = 45
n_img               = 48

start               = 3
duration            = 1

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
color_trial         = ['dodgerblue', np.array([212, 170, 0])/255]

# set time window
time_window = 5

# # computations
computation                = ['log fit']

# fit curve for recovery of adaptation initial parameter values
t1_plot     = np.linspace(min(tempCond), max(tempCond), 1000)
p0          = [1, 0]

# initiate dataframes
data                = np.zeros((len(trials), n_layer, len(tempCond), n_img, nt))
# data_one_pulse      = np.zeros((len(trials), n_layer, len(tempCond), n_img, nt))
# data_two_pulse      = np.zeros((len(trials), n_layer, len(tempCond), n_img, nt))

metric              = np.zeros((len(trials), n_layer, len(tempCond), n_img))

dynamics_log        = np.zeros((len(trials), n_layer, n_img, len(t1_plot)))
dynamics_fit        = np.zeros((len(trials), n_layer, n_img))

ratio_lin_log       = np.zeros((len(trials), n_layer, n_img))

adaptation_avg      = np.zeros((len(trials), n_layer, n_img))

# plot model activations
lbls = []
for iL in range(n_layer):
    for iT, trial in enumerate(trials):
        for i, datasetstim in enumerate(dataset_stim):

            # retrieve data
            if population == None:
                if trained:
                    temp = np.load(root_data + datasetstim + '/avgs/prednet_' + trial + '_' + output_mode + str(iL+1) + '_actvs_' + dataset + '.npy')
                else:
                    temp = np.load(root_data + datasetstim + '/avgs/prednet_' + trial + '_' + output_mode + str(iL+1) + '_actvs_random.npy')
            else:
                if trained:
                    temp = np.load(root_data + datasetstim + '/avgs/prednet_' + trial + '_' + output_mode + str(iL+1) + population + '_actvs_' + dataset + '.npy')
                else:
                    temp = np.load(root_data + datasetstim + '/avgs/prednet_' + trial + '_' + output_mode + str(iL+1) + population + '_actvs_random.npy')

            # store data
            if i == 0:
                data[iT, iL, :, :int(n_img/2), :] = temp
            else:
                data[iT, iL, :, int(n_img/2):, :] = temp

        # obtain DATA
        for iC in range(len(tempCond)):

            # compute RS
            for iImg in range(n_img):

                # normalize data
                if iC > 2:
                    data_ISI6 = data[iT, iL, iC, iImg, :] - np.mean(data[iT, iL, -1, iImg, start-1])
                else:
                    data_ISI6 = data[iT, iL, -1, iImg, :] - np.mean(data[iT, iL, -1, iImg, start-1])

                data_ISI = data[iT, iL, iC, iImg, :] - np.mean(data[iT, iL, iC, iImg, start-1])

                # compute first and second pulse
                first_pulse = np.zeros(nt)
                first_pulse[:start + duration + time_window] = data_ISI6[:start + duration + time_window]
                second_pulse = data_ISI - first_pulse

                # compute response
                AUC1 = np.max(first_pulse[start:start+duration+time_window])
                AUC2 = np.max(second_pulse[start+tempCond[iC]+duration:start+tempCond[iC]+duration+time_window])

                # compute metric
                metric[iT, iL, iC, iImg] = AUC2/AUC1

                if (iL == 1) & (iT == 1) & (iImg == 8) & (iC == 3):
                    
                    # plot
                    plt.plot(data[iT, iL, iC, iImg, :], color='black', label='data')
                    plt.plot(first_pulse, color='grey', label='first_pulse')
                    plt.plot(second_pulse, color='red', label='second pulse')


                    plt.axvline(start, linestyle='dashed', color='grey')
                    plt.axvspan(start, start+duration+time_window, color='silver')
                    plt.axvspan(start+tempCond[iC], start+tempCond[iC]+duration+time_window, color='silver')

                    # print metric
                    print(AUC1)
                    print(AUC2)
                    print(metric[iT, iL, iC, iImg])

                    # savefig
                    plt.legend()
                    plt.savefig(root+'visualization/model/compute_recovery')

            # compute line fit
            for iImg in range(n_img):

                # fit curve - log
                popt, _ = curve_fit(OF_dynamics_log, tempCond, metric[iT, iL, :, iImg], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                dynamics_log[iT, iL, iImg, :] = OF_dynamics_log(t1_plot, *popt)

                pred = OF_dynamics_log(tempCond, *popt)
                dynamics_fit[iT, iL, iImg] = r_squared(metric[iT, iL, :, iImg], pred)

        # compute average
        adaptation_avg[iT, iL, :] = np.mean(metric[iT, iL, :, :], 0)

######################## VISUALIZE
# plot regression
# plot_broadband_all(data, trials_lbls, start, tempCond, duration, n_layer, n_img, population, dataset, output_mode, color_trial, root_data, root_vis)
plot_broadband(data, None, None, trials_lbls, start, tempCond, duration, n_img, color_trial, root_vis)

plot_dynamics(dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root_vis)
plot_dynamics_metric(dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root_vis)

# plot_regression_CEandSC(ratio_lin_log, metric, CEandSC_values, n_layer, color_layer, n_img, model, root_vis)

######################## STATISTICS
stats_trial(metric, n_layer)
