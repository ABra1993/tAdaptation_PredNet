import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import pandas as pd
import math
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.utils import resample

# import functions
from neural_data_visualize_utils import *
from Fig5_neural_data_visualize_twopulse_utils import *

# directory containing neural data
root            = '/home/amber/OneDrive/code/prednet_Brands2024_git/'

root_data       = '/home/amber/Documents/prednet_Brands2024/data/ECoG/'
root_channels   = '/home/amber/OneDrive/code/nAdaptation_ECoG_git/data_subjects/'

# visual areas
visual_area = ['V1-V3', 'VOTC', 'LOTC']

# responsive electrodes
electrodes_selection = pd.read_csv(root_channels + 'electrodes_visuallyResponsive_manuallyAssigned.txt', sep=' ', header=0)
electrodes_selection = electrodes_selection[electrodes_selection.subject != 'sub-p11']
electrodes_selection.reset_index(inplace=True, drop=True)
n_electrodes = len(electrodes_selection)

# extract electrode indices per visual area (i.e. V1-V3, LOTC, VOTC)
VA_name_idx_temp = {}
for i in range(n_electrodes):
    VA_name_current = electrodes_selection.loc[i, 'varea']
    if VA_name_current not in VA_name_idx_temp:
        VA_name_idx_temp[VA_name_current] = [i]
    else:
        VA_name_idx_temp[VA_name_current].append(i)
VA_name_idx = {}
VA_name_idx = {k: VA_name_idx_temp[k] for k in visual_area}
print(VA_name_idx, '\n')

# subjects
subjects = electrodes_selection.subject.unique()
print(subjects)

# plot per visual area
n_electrodes_per_area   = np.zeros(len(visual_area), dtype=int)

# trial type
trials          = np.array(['twopulse_repeat', 'twopulse_nonrepeat_same'])
trials_lbls     = np.array(['repeat', 'nonrepeat (same)'])
color_trial     = ['dodgerblue', np.array([212, 170, 0])/255]


# import timepoints
t                           = np.loadtxt(root + 'variables/t.txt', dtype=float)
timepoints_onepulse         = np.loadtxt(root + 'variables/timepoints_onepulse.txt', dtype=int)
timepoints_twopulse         = np.loadtxt(root + 'variables/timepoints_twopulse.txt', dtype=int)
time_window                 = np.loadtxt(root + 'variables/time_window_twopulse.txt', dtype=int)
tempCond                    = np.loadtxt(root + 'variables/cond_temp.txt', dtype=int)
n_tempCond                  = len(tempCond)

print('Time window: ', str(t[timepoints_onepulse[0, 0]+time_window]))

# fit curve for recovery of adaptation initial parameter values
t1_plot     = np.linspace(min(tempCond), max(tempCond), 1000)
p0          = [0, 1]

t_end       = 750
lw          = 1

# axes settings
# color = ['black', np.array([136, 204, 238])/255, np.array([68, 170, 153])/255, np.array([17, 119, 51])/255]
color = ['#DBA507', '#1BBC9B', '#00796B']
lw = 1

# initiate dataframes for linear/nonlinear processing
computation                = ['linear', 'log']

# bootstrap data
bootstrapped = False

# determine confidence interval for plotting
CI                  = 68
CI_low              = 50 - (0.5*CI)
CI_high             = 50 + (0.5*CI)
B_repetitions       = 1000

# initiate dataframes - bootstrapped
if bootstrapped:

    data                = np.zeros((len(visual_area), len(trials), len(tempCond), B_repetitions, len(t)))
    data_onepulse       = np.zeros((len(visual_area), len(trials), B_repetitions, len(t)))
    data_twopulse       = np.zeros((len(visual_area), len(trials), len(tempCond), B_repetitions, len(t)))

    metric              = np.zeros((len(visual_area), len(trials), len(tempCond), B_repetitions))

    dynamics_lin        = np.zeros((len(visual_area), len(trials), B_repetitions, len(t1_plot)))
    dynamics_log        = np.zeros((len(visual_area), len(trials), B_repetitions, len(t1_plot)))
    dynamics_fit        = np.zeros((len(visual_area), len(trials), B_repetitions, len(computation)))

    adaptation_avg      = np.zeros((len(visual_area), len(trials), B_repetitions))

else:

    data                = list()
    data_onepulse       = list()
    data_twopulse       = list()

    metric              = list()

    dynamics_lin        = list()
    dynamics_log        = list()
    dynamics_fit        = list()

    adaptation_avg      = list()

# load data
data_all_onepulse = np.load(root_data + 'data_onepulse.npy')
data_all_onepulse = data_all_onepulse[3, :, :] # 4th temporal condition of 134 ms
print(data_all_onepulse.shape)

# retreive
count = 0
for key, values in VA_name_idx.items():

    # count
    n_electrodes_per_area[count] = len(values)

    # initiate dataframes
    if bootstrapped == False:

        data_VA                 = np.zeros((len(trials), len(tempCond), n_electrodes_per_area[count], len(t)))
        data_onepulse_VA        = np.zeros((len(trials), n_electrodes_per_area[count], len(t)))
        data_twopulse_VA        = np.zeros((len(trials), len(tempCond), n_electrodes_per_area[count], len(t)))

        metric_VA               = np.zeros((len(trials), len(tempCond), n_electrodes_per_area[count]))

        dynamics_lin_VA         = np.zeros((len(trials), n_electrodes_per_area[count], len(t1_plot)))
        dynamics_log_VA         = np.zeros((len(trials), n_electrodes_per_area[count], len(t1_plot)))
        dynamics_fit_VA         = np.zeros((len(trials), n_electrodes_per_area[count], len(computation)))

        adaptation_avg_VA       = np.zeros((len(trials), n_electrodes_per_area[count]))

    # retrieve data
    for iT, trial in enumerate(trials):

        # load data
        data_all = np.load(root_data + '/data_' + trial + '.npy')
        # print(data_all.shape)

        # obtain DATA
        if bootstrapped:
            for iB in range(B_repetitions):

                # draw random sample
                boot = resample(values, replace=True, n_samples=n_electrodes)

                # compute onepulse
                data_mean_onepulse = np.zeros((len(boot), len(t)))
                for l in range(len(boot)):
                    data_mean_onepulse[l, :] = data_all_onepulse[boot[l], :]

                # compute mean
                data_onepulse[count, iT, iB, :] = np.mean(data_mean_onepulse, 0)

                for iC in range(len(tempCond)):

                    # retrieve samples
                    data_mean = np.zeros((len(boot), len(t)))
                    for l in range(len(boot)):
                        data_mean[l, :] = data_all[iC, boot[l], :]

                    # compute mean
                    data[count, iT, iC, iB, :] = np.mean(data_mean, 0)

                    # compute isolated second pulse
                    data_twopulse[count, iT, iC, iB, :] = data[count, iT, iC, iB, :] - data_onepulse[count, iT, iB, :]
                    
                    # compute AUC
                    metric[count, iT, iC, iB] = np.max(data_twopulse[count, iT, iC, iB, timepoints_twopulse[iC, 2]:timepoints_twopulse[iC, 2]+time_window])/np.max(data_onepulse[count, iT, iB, timepoints_onepulse[3, 0]:timepoints_twopulse[3, 1]+time_window])
        
        else:

            for iC, cond in enumerate(tempCond):

                # select data
                data_VA[iT, iC, :, :] = data_all[iC, values, :]
                data_onepulse_VA[iT, :, :] = data_all_onepulse[values, :]

                # compute isolated second pulse
                data_twopulse_VA[iT, iC, :, :] = data_VA[iT, iC, :, :] - data_onepulse_VA[iT, :, :]

                # compute metric
                for iV in range(len(values)):
                    metric_VA[iT, iC, iV] = np.max(data_twopulse_VA[iT, iC, iV, timepoints_twopulse[iC, 2]:timepoints_twopulse[iC, 2]+time_window])/np.max(data_onepulse_VA[iT, iV, timepoints_onepulse[3, 0]:timepoints_twopulse[3, 1]+time_window])

        # compute FIT
        if bootstrapped:

            # compute line fit
            for iB in range(B_repetitions):

                # fit curve - lin
                popt, _ = curve_fit(OF_dynamics_linear, tempCond, metric[count, iT, :, iB], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                dynamics_lin[count, iT, iB, :] = OF_dynamics_linear(t1_plot, *popt)

                pred = OF_dynamics_linear(tempCond, *popt)
                dynamics_fit[count, iT, iB, 0] = r_squared(metric[count, iT, :, iB], pred)

                # fit curve - log
                popt, _ = curve_fit(OF_dynamics_log, tempCond, metric[count, iT, :, iB], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                dynamics_log[count, iT, iB, :] = OF_dynamics_log(t1_plot, *popt)

                pred = OF_dynamics_log(tempCond, *popt)
                dynamics_fit[count, iT, iB, 1] = r_squared(metric[count, iT, :, iB], pred)

            # compute average adaptation
            adaptation_avg[count, iT, :] = np.mean(metric[count, iT, :, :], 0)

        else:

            # compute line fit
            for iV in range(len(values)):

                # fit curve - lin
                popt, _ = curve_fit(OF_dynamics_linear, tempCond, metric_VA[iT, :, iV], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                dynamics_lin_VA[iT, iV, :] = OF_dynamics_linear(t1_plot, *popt)

                pred = OF_dynamics_linear(tempCond, *popt)
                dynamics_fit_VA[iT, iV, 0] = r_squared(metric_VA[iT, :, iV], pred)

                # fit curve - log
                popt, _ = curve_fit(OF_dynamics_log, tempCond, metric_VA[iT, :, iV], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
                dynamics_log_VA[iT, iV, :] = OF_dynamics_log(t1_plot, *popt)

                pred = OF_dynamics_log(tempCond, *popt)
                dynamics_fit_VA[iT, iV, 1] = r_squared(metric_VA[iT, :, iV], pred)

            # compute average adaptation
            adaptation_avg_VA[iT, :] = np.mean(metric_VA[iT, :, :], 0)

    # add to dataframe
    if bootstrapped == False:

        data.append(data_VA)
        data_onepulse.append(data_onepulse_VA)
        data_twopulse.append(data_twopulse_VA)

        metric.append(metric_VA)

        dynamics_lin.append(dynamics_lin_VA)
        dynamics_log.append(dynamics_log_VA)
        dynamics_fit.append(dynamics_fit_VA)

        adaptation_avg.append(adaptation_avg_VA)

    # increment count
    count+=1

######################## VISUALIZE
plot_broadband(data, bootstrapped, tempCond, trials_lbls, visual_area, n_electrodes, n_electrodes_per_area, t, timepoints_twopulse, color_trial, root)

plot_dynamics(metric, adaptation_avg, dynamics_lin, dynamics_log, visual_area, trials_lbls, bootstrapped, computation, tempCond, CI_low, CI_high, color, color_trial, n_electrodes_per_area, t1_plot, root)
plot_dynamics_metric(metric, adaptation_avg, dynamics_lin, dynamics_log, visual_area, trials_lbls, bootstrapped, computation, tempCond, CI_low, CI_high, color, color_trial, n_electrodes_per_area, t1_plot, root)

######################## SAVE
stats_trial(adaptation_avg, visual_area)
