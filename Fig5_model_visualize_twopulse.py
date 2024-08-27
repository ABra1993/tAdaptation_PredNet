import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from neural_data_visualize_utils import *
from Fig5_model_visualize_twopulse_utils import *

# set root
root            = '/home/amber/OneDrive/code/prednet_Brands2024_git/'

root_data       = '/home/amber/Documents/prednet_Brands2024/data/model/Lotter2017/'
root_sumMet     = '/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/'

# stimulus dataset
dataset_stim = ['set1', 'set2']

# select model
model = 'Lotter2017'

# set training data
dataset = 'KITTI'

# trial types
# trials = np.array(['twopulse_repeat', 'twopulse_nonrepeat_same', 'twopulse_nonrepeat_diff'])
# trials_lbls = ['repetition', 'alternation (within)', 'alternation (between)']
# color_trial         = ['#DBA507', '#1BBC9B', '#00796B']

trials = np.array(['twopulse_repeat', 'twopulse_nonrepeat_same'])
trials_lbls = ['repetition', 'alternation (within)']

# set trained or untrained
trained = True

# experimental info
nt                  = 45
n_img               = 48

start               = 4
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
time_window = 3

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
# for iL in range(3, 4):

    # # import trials onepulse
    # data_onepulse = np.zeros((len(tempCond), n_img, nt))
    # for i, datasetstim in enumerate(dataset_stim):

    #     # retrieve data
    #     if population == None:
    #         if trained:
    #             temp = np.load(root_data + datasetstim + '/avgs/prednet_onepulse_' + output_mode + str(iL+1) + '_actvs_' + dataset + '.npy')
    #         else:
    #             temp = np.load(root_data + datasetstim + '/avgs/prednet_onepulse_' + output_mode + str(iL+1) + '_actvs_random.npy')
    #     else:
    #         if trained:
    #             temp = np.load(root_data + datasetstim + '/avgs/prednet_onepulse_' + output_mode + str(iL+1) + population + '_actvs_' + dataset + '.npy')
    #         else:
    #             temp = np.load(root_data + datasetstim + '/avgs/prednet_onepulse_' + output_mode + str(iL+1) + population + '_actvs_random.npy')
        
    #     # store data
    #     if i == 0:
    #         data_onepulse[:, :int(n_img/2), :] = temp
    #     else:
    #         data_onepulse[:, int(n_img/2):, :] = temp

    # analyse twopulse trials
    for iT, trial in enumerate(trials):

        # import trials onepulse
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

                # avg = np.mean(data[iT, iL, iC, iImg, :start])
                # data[iT, iL, iC, iImg, :] = data[iT, iL, iC, iImg, :] - avg


                twopulse = np.zeros(nt)
                twopulse[:start + duration + time_window] = data[iT, iL, iC, iImg, :start + duration + time_window]
                onepulse = data[iT, iL, iC, iImg, :] - twopulse

                # if (iL == 0)& (iC == 4) & (iImg == 1):
                    


                #     # plt.plot(data[iT, iL, iC, iImg, :])
                #     # plt.plot(data[iT, iL, 5, iImg, :])

                #     plt.plot(onepulse, label='isolated', color=color_trial[iT])
                #     plt.plot(twopulse, label='raw', color=color_trial[iT])
                #     print(np.max(onepulse))
                #     print(np.max(twopulse))
                #     plt.legend()

                
                # compute response
                # data_one_pulse[iT, iL, iC, iImg, :] = data_onepulse[0, iImg, :] - np.mean(data_onepulse[0, iImg, 3:], 0)
                # print(onepulse)
                AUC1 = np.max(onepulse[start:start+duration+time_window+tempCond[iC]])
                # print(AUC1)

                # # compute isolated second pulse
                # data_current = data[iT, iL, iC, iImg, :]
                # data_two_pulse[iT, iL, iC, iImg, :] = data[iT, iL, iC, iImg, :] - data_onepulse[0, iImg, :]
                # data_two_pulse[iT, iL, iC, iImg, :start+duration+tempCond[iC]+1] = np.zeros(start+duration+tempCond[iC]+1)

                AUC2 = np.max(twopulse[start:start+duration+time_window])
                # print(AUC2)

                # compute metric
                metric[iT, iL, iC, iImg] = AUC1/AUC2

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
plot_broadband(data, None, None, trials_lbls, start, tempCond, duration, n_img, color_trial, root)
plot_dynamics(dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root)
plot_dynamics_metric(dynamics_log, adaptation_avg, metric, tempCond, trials, t1_plot, n_layer, color_layer, color_trial, n_img, computation, output_mode, model, root)

######################## STATISTICS
stats_trial(adaptation_avg, n_layer)
