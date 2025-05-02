import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from __model_visualize_onepulse_utils import *

# set root
root        = '' ## ADD home directory 

# select model
model = 'Lotter2017'
# model = 'Kirubeswaran2023'

# set directory's
if model == 'Lotter2017':
    root_data       = '' ## ADD location of activations
    root_vis        = '' ## ADD location to store figures
elif model == 'Kirubeswaran2023':
    root_data       = '' ## ADD location of activations
    root_vis        = '' ## ADD location to store figures

# set training data
if model == 'Lotter2017':
    dataset = 'KITTI'
elif model == 'Kirubeswaran2023':
    dataset = 'FPSI'

# set trained or untrained
trained = True

# experimental info
nt                  = 45
n_img               = 1600
start               = 3
stim_duration       = 32

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

CE = np.loadtxt(root + '/imgs_statistics/Groen2013/CE.txt', delimiter=',')
SC = np.loadtxt(root + '/imgs_statistics/Groen2013/SC.txt', delimiter=',')

CEandSC_values[:, 0] = CE
CEandSC_values[:, 1] = SC

# plot settings
color_tempCond      = ['#9BD2E1', '#81C4E7', '#7EB2E4', '#9398D2', '#9D7DB2', '#906388']
color_layer         = ['#69B190', '#549EB3', '#4E79C5', '#6F4C9B']

# computations
computation         = ['linear fit', 'log fit']

# initiate dataframes
data                = np.zeros((n_layer, n_img, nt))
metric              = np.zeros((n_layer, n_img))

ratio_lin_log       = np.zeros((n_layer, n_img))
ratio_trans_sust    = np.zeros((n_layer, n_img))

# time window ranges (longest duration)
range_trans     = [start, start+2]
range_sust      = [start+stim_duration-20, start+stim_duration]

# plot model activations
lbls = []
for iL in range(n_layer):

        # retrieve data
        if population == None:
            if trained:
                temp = np.load(root_data + 'avgs/prednet_onepulse_' + output_mode + str(iL+1) + '_actvs_trained.npy')
            else:
                temp = np.load(root_data + 'avgs/prednet_onepulse_' + output_mode + str(iL+1) + '_actvs_random.npy')
        else:
            if trained:
                temp = np.load(root_data + 'avgs/prednet_onepulse_' + output_mode + str(iL+1) + population + '_actvs_' + dataset + '.npy')
            else:
                temp = np.load(root_data + 'avgs/prednet_onepulse_' + output_mode + str(iL+1) + population + '_actvs_random.npy')

        # store data
        data[iL, :, :] = temp

        # compute TRANS:SUST ratio
        for iImg in range(n_img):

            # compute T:S ratio
            data_current = data[iL, iImg, :]

            # compute peak and transient
            trans = np.max(data_current[range_trans[0]:range_trans[1]])
            sust = np.mean(data_current[range_sust[0]:range_sust[1]])
            ratio_trans_sust[iL, iImg] = trans/sust

####################### VISUALIZE
#################################

# FIG 8D
plot_regression_CEandSC_L1(ratio_lin_log, ratio_trans_sust, CEandSC_values, n_layer, color_layer, n_img, model, root_vis)
