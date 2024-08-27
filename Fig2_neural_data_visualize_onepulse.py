import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import pandas as pd
import math
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.utils import resample
from scipy import stats
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# import functions
from neural_data_visualize_utils import *
from Fig2_neural_data_visualize_onepulse_utils import *

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
# print(VA_name_idx, '\n')

# subjects
subjects = electrodes_selection.subject.unique()
# print(subjects)

# plot per visual area
n_electrodes_per_area   = np.zeros(len(visual_area), dtype=int)

# import timepoints
t                           = np.loadtxt(root + 'variables/t.txt', dtype=float)
timepoints_onepulse         = np.loadtxt(root + 'variables/timepoints_onepulse.txt', dtype=int)
timepoints_twopulse         = np.loadtxt(root + 'variables/timepoints_twopulse.txt', dtype=int)
time_window                 = np.loadtxt(root + 'variables/time_window_onepulse.txt', dtype=int)
tempCond                    = np.loadtxt(root + 'variables/cond_temp.txt', dtype=int)
n_tempCond                  = len(tempCond)

print('Time window: ', str(t[timepoints_onepulse[0, 0]+time_window]))

# initiate dataframes for linear/nonlinear processing
computation                = ['linear', 'log']

# bootstrap procedure
bootstrapped = False

# determine confidence interval for plotting
CI                  = 68
CI_low              = 50 - (0.5*CI)
CI_high             = 50 + (0.5*CI)
B_repetitions       = 1000

# fit curve for recovery of adaptation initial parameter values
t1_plot             = np.linspace(min(tempCond), max(tempCond), 1000)
p0                  = [1, 0]

# plot settings
color = ['#DDCC77', '#999933', '#117733']

# load data
data_all = np.load(root_data + 'data_onepulse.npy')
print(data_all.shape)

# initiate dataframes - bootstrapped
if bootstrapped:

    data                = np.zeros((len(visual_area), len(tempCond), B_repetitions, len(t)))
    metric              = np.zeros((len(visual_area), len(tempCond), B_repetitions))

    dynamics_lin        = np.zeros((len(visual_area), B_repetitions, len(t1_plot)))
    dynamics_log        = np.zeros((len(visual_area), B_repetitions, len(t1_plot)))
    dynamics_fit        = np.zeros((len(visual_area), B_repetitions, len(computation)))

    ratio_lin_log       = np.zeros((len(visual_area), B_repetitions))
    ratio_trans_sust    = np.zeros((len(visual_area), B_repetitions))

else:

    data                = list()
    metric              = list()

    dynamics_lin        = list()
    dynamics_log        = list()
    dynamics_fit        = list()

    ratio_lin_log       = list()
    ratio_trans_sust    = list()

# time window ranges (longest duration)
range_trans     = [timepoints_onepulse[-1, 0], timepoints_onepulse[-1, 0]+120]
range_sust      = [timepoints_onepulse[-1, 1]-140, timepoints_onepulse[-1, 1]]

# retreive
count = 0
for key, values in VA_name_idx.items():

    # count
    n_electrodes_per_area[count] = len(values)

    # initiate dataframes
    if bootstrapped == False:

        data_VA                 = np.zeros((len(tempCond), n_electrodes_per_area[count], len(t)))
        metric_VA               = np.zeros((len(tempCond), n_electrodes_per_area[count]))

        dynamics_lin_VA         = np.zeros((n_electrodes_per_area[count], len(t1_plot)))
        dynamics_log_VA         = np.zeros((n_electrodes_per_area[count], len(t1_plot)))
        dynamics_fit_VA         = np.zeros((n_electrodes_per_area[count], len(computation)))

        ratio_lin_log_VA        = np.zeros(n_electrodes_per_area[count])
        ratio_trans_sust_VA     = np.zeros(n_electrodes_per_area[count])

    # obtain DATA
    if bootstrapped:
        for iB in range(B_repetitions):

            # draw random sample
            boot = resample(values, replace=True, n_samples=n_electrodes_per_area[count])
            
            for iC in range(len(tempCond)):

                # retrieve samples
                data_mean = np.zeros((len(boot), len(t)))
                for l in range(len(boot)):
                    data_mean[l, :] = data_all[iC, boot[l], :]
                
                # compute mean
                data[count, iC, iB, :] = np.mean(data_mean, 0)

    else:

        for iC in range(len(tempCond)):
            data_VA[iC, :, :] = data_all[iC, values, :]

    # compute TRANS:SUST ratio
    if bootstrapped:

        for iB in range(B_repetitions):

            # compute T:S ratio
            data_current = data[count, -1, iB, :]
            
            # compute peak and transient
            trans = np.max(data_current[range_trans[0]:range_trans[1]])
            sust = np.mean(data_current[range_sust[0]:range_sust[1]])
            ratio_trans_sust[count, iB] = trans/sust

    else:

        for iV in range(len(values)):

            # compute T:S ratio
            data_current = data_VA[-1, iV, :]
            
            # compute peak and transient
            trans = np.max(data_current[range_trans[0]:range_trans[1]])
            sust = np.mean(data_current[range_sust[0]:range_sust[1]])
            ratio_trans_sust_VA[iV] = trans/sust

    # compute FIT
    if bootstrapped:

        # compute metric over ALL stim. dur.
        for iC in range(len(tempCond)):
            for iB in range(B_repetitions):

                # select and normalize data
                data_value = data[count, iC, iB, timepoints_onepulse[iC, 0]:timepoints_onepulse[iC, 0]+time_window]

                # compute AUC
                metric[count, iC, iB] = np.trapz(data_value)/np.trapz(data[count, -1, iB, timepoints_onepulse[-1, 0]:timepoints_onepulse[-1, 0]+time_window])

        # compute line fit
        for iB in range(B_repetitions):

            # fit curve - lin
            popt, _ = curve_fit(OF_dynamics_linear, tempCond, metric[count, :, iB], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
            dynamics_lin[count, iB, :] = OF_dynamics_linear(t1_plot, *popt)

            pred = OF_dynamics_linear(tempCond, *popt)
            dynamics_fit[count, iB, 0] = r_squared(metric[count, :, iB], pred)

            # fit curve - log
            popt, _ = curve_fit(OF_dynamics_log, tempCond, metric[count, :, iB], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
            dynamics_log[count, iB, :] = OF_dynamics_log(t1_plot, *popt)

            pred = OF_dynamics_log(tempCond, *popt)
            dynamics_fit[count, iB, 1] = r_squared(metric[count, :, iB], pred)

            # compute ratio
            ratio_lin_log[count, iB] = dynamics_fit[count, iB, 0]/dynamics_fit[count, iB, 1]

    else:

        # compute bootstrapped data
        for iV in range(len(values)):

            for iC in range(len(tempCond)):

                # select and normalize data
                data_value = data_VA[iC, iV, timepoints_onepulse[iC, 0]:timepoints_onepulse[iC, 0]+time_window]

                # compute AUC
                metric_VA[iC, iV] = np.trapz(data_value)/np.trapz(data_VA[-1, iV, timepoints_onepulse[iC, 0]:timepoints_onepulse[iC, 0]+time_window])

        # compute line fit
        for iV in range(len(values)):

            # fit curve - lin
            popt, _ = curve_fit(OF_dynamics_linear, tempCond, metric_VA[:, iV], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
            dynamics_lin_VA[iV, :] = OF_dynamics_linear(t1_plot, *popt)

            pred = OF_dynamics_linear(tempCond, *popt)
            dynamics_fit_VA[iV, 0] = r_squared(metric_VA[:, iV], pred)

            # fit curve - log
            popt, _ = curve_fit(OF_dynamics_log, tempCond, metric_VA[:, iV], p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
            dynamics_log_VA[iV, :] = OF_dynamics_log(t1_plot, *popt)

            pred = OF_dynamics_log(tempCond, *popt)
            dynamics_fit_VA[iV, 1] = r_squared(metric_VA[:, iV], pred)

            # compute ratio
            ratio_lin_log_VA[iV] = dynamics_fit_VA[iV, 0]/dynamics_fit_VA[iV, 1]

        # add to dataframe
        if bootstrapped == False:

            data.append(data_VA)
            metric.append(metric_VA)

            dynamics_lin.append(dynamics_lin_VA)
            dynamics_log.append(dynamics_log_VA)
            dynamics_fit.append(dynamics_fit_VA)

            ratio_lin_log.append(ratio_lin_log_VA)
            ratio_trans_sust.append(ratio_trans_sust_VA)


    # increment count
    count+=1

######################## VISUALIZE
plot_broadband(data_all, data, bootstrapped, tempCond, visual_area, n_electrodes, n_electrodes_per_area, t, timepoints_onepulse, range_trans, range_sust, color, root)      # Fig. 2A
plot_dynamics(metric, dynamics_lin, dynamics_log, visual_area, bootstrapped, computation, tempCond, CI_low, CI_high, color, n_electrodes_per_area, t1_plot, root)           # Fig. 2B
plot_dynamics_metric(dynamics_fit, ratio_lin_log, visual_area, n_electrodes_per_area, bootstrapped, computation, CI_low, CI_high, color, root)                              # Fig. 2C
plot_regression(visual_area, color, ratio_trans_sust, ratio_lin_log, bootstrapped, CI_low, CI_high, n_electrodes_per_area, root)                                            # Fig. 2D

# ######################## PERFORM STATISTICS
stats_lin_log(visual_area, bootstrapped, B_repetitions, dynamics_fit)
stats_regression(ratio_lin_log, bootstrapped, ratio_trans_sust)
