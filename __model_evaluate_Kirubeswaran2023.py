import os
import numpy as np
from PIL import Image, ImageCms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import prednet_pyTorch
import argparse
import datetime
from tqdm import tqdm
from distutils.util import strtobool
# from dataset import ImageListDataset
# from corr_wise import CorrWise
import glob
import matplotlib.pyplot as plt

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# set folders
root        = '' ## ADD home directory
root_stim   = '' ## ADD folder containing the stimuli
root_save   = '' ## ADD folder to which the activations are saved

# test dataset
test_set        = ['set1', 'set2']

######################## ANALYSE ------- DATASET (init = 5)
# train_set = 'KITTI'
# train_set = 'WT_AMS'
# train_set = 'WT_VEN'
# train_set = 'WT_WL'

# train_sets = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']

############################ ANALYSE ------- FPS (init = 3)
# train_sets = ['KITTI_fps3', 'KITTI_fps6', 'KITTI_fps12']
# train_sets = ['WT_AMS_fps3', 'WT_AMS_fps6', 'WT_AMS_fps12']
# train_sets = ['WT_VEN_fps3', 'WT_VEN_fps6', 'WT_VEN_fps12']
# train_sets = ['WT_WL_fps3', 'WT_WL_fps6', 'WT_WL_fps12']

########################### ANALYSE ------- LOSS
# train_sets = ['KITTI_Lnull', 'KITTI_Lall']
# train_sets = ['WT_AMS_Lnull', 'WT_AMS_Lall']
# train_sets = ['WT_VEN_Lnull', 'WT_VEN_Lall']
# train_sets = ['WT_WL_Lnull', 'WT_WL_Lall']

train_sets = ['WT_AMS_fps3', 'WT_AMS_fps6', 'WT_AMS_fps12']

# # plot predictions
# plot_prediction = False
# plot_start      = 0
# plot_end        = 10

# fontsizes 
fontsize_title          = 20
fontsize_legend         = 12
fontsize_label          = 10
fontsize_tick           = 10

def main():

    ########################## NETWORK SETTINGS
    
    # evaluation settings
    batch_size          = 1
    n_img               = 24
 
    # stimulus settings
    tempCond            = [1, 2, 4, 8, 16, 32]
    nt                  = 45

    # network settings
    n_layer             = 4
    channels            = [3, 48, 96, 192]
    trained             = True

    # trial types
    trials = np.array(['onepulse', 'twopulse_repeat', 'twopulse_nonrepeat_same'])
    # trials = np.array(['twopulse_repeat', 'twopulse_nonrepeat_same'])

    # output mode
    output_modes = ['E']
    for output_mode in output_modes:
        assert output_mode in ['E', 'R', 'A', 'Ahat']

    # iterate over training dataset and retrieve activations
    for train_set in train_sets:

        # select folder
        if ('fps' in train_set):
            analyse = 'fps'
        elif ('Lnull' in train_set) | ('Lall' in train_set):
            analyse = 'loss'
        else:
            analyse = 'dataset'

        # number of random network initializations
        if analyse == 'dataset':
            init                = 5
        else:
            init                = 3

        # summarize
        print('trials: '.ljust(25), trials)
        print('output mode: '.ljust(25), output_modes)
        print('trained: '.ljust(25), trained)
        print('training set: '.ljust(25), train_set)

        # root dataset
        if 'KITTI' in train_set:
            dataset = 'KITTI'
        elif 'WT_AMS' in train_set:
            dataset = 'WT_AMS'
        elif 'WT_VEN' in train_set:
            dataset = 'WT_VEN'
        elif 'WT_WL' in train_set:
            dataset = 'WT_WL'

        for iInit in range(init):

            # initiate model
            net = prednet_pyTorch.PredNet(channels, device=device).to(device)
            net.eval()

            # load weights
            if train_set != 'random':
                if analyse == 'dataset':
                    initmodel = root + 'weights/' + analyse + '/' + train_set + str(iInit+1) + '.pth'
                else:
                    initmodel = root + 'weights/' + analyse + '/' + train_set + '_' + str(iInit+1) + '.pth'
                net.load_state_dict(torch.load(initmodel))

            ########################## INFERENCE

            # initiate dataframe
            # metrics_avg = np.zeros((len(tempCond), n_img, nt, 4)) # 4 = number of layers

            # torch.Size([1, 6, 120, 160])
            # torch.Size([1, 96, 60, 80])
            # torch.Size([1, 192, 30, 40])
            # torch.Size([1, 384, 15, 20])

            # sizes = [torch.Size([1, 6, 120, 160]), torch.Size([1, 96, 60, 80]), torch.Size([1, 192, 30, 40]), torch.Size([1, 384, 15, 20])]

            for output_mode in output_modes:

                for test_set_current in test_set:

                    # print progress
                    print('Output mode: ', output_mode)

                    for trial in trials:

                        # initiate dataframe
                        metrics_avg = list()
                        for iL in range(n_layer):
                            temp = np.zeros((len(tempCond), n_img, nt))
                            metrics_avg.append(temp)

                        # print progress
                        print('Trial: ', trial)

                        # load stimuli
                        X_test = np.load(root_stim + test_set_current +  '/stimuli_' + trial + '.npy')
                        print(X_test.shape)

                        # X_test = np.load(root_stim +  'datasets/stimuli_' + trial + '_WT_VEN.npy')
                        # print(X_test.shape)

                        # # set tempCond to be tested
                        # if plot_prediction:
                        #     range_tempCond = [5, len(tempCond)]
                        # else:
                        #     range_tempCond = [0, len(tempCond)]

                        # for iC in range(range_tempCond[0], range_tempCond[1]):
                        for iT in range(len(tempCond)):

                            # print progress
                            print('Temp cond: ', tempCond[iT])

                            # load data with batch size of 1
                            data_loader = DataLoader(X_test[iT, :, :, :, :, :], batch_size=batch_size, shuffle=False)

                            for i, data in enumerate(tqdm(data_loader, unit="batch")):
                                
                                # # visualize prediction
                                # if plot_prediction:

                                #     if i != 20:
                                #         continue

                                #     # initiate plot
                                #     _, axs = plt.subplots(3, plot_end-plot_start+1, figsize=(8, 3))

                                #     for t in range(plot_start, plot_end+1):
                                #         axs[0, t-plot_start].set_title('t=' + str(t+1), fontsize=fontsize_label, rotation=45)
                                #         axs[0, t-plot_start].imshow(np.transpose(data[:, t, :, :, :].detach().cpu().numpy().squeeze(), (1, 2, 0)))

                                # iterate over timepoints
                                for t in range(nt-1): # last index 44

                                    # select current data
                                    current_data = data[:, :t+2, :, :, :]
                                    # print('Data: ', current_data.shape)

                                    with torch.no_grad():
                                        with torch.cuda.amp.autocast(enabled=True):
                                            
                                            # predict
                                            _, _, _, _, E_seq = net(current_data.to(device))

                                            # save metric
                                            for iL in range(n_layer):
                                                metrics_avg[iL][iT, i, t+1] = np.mean(E_seq[iL].detach().cpu().numpy().squeeze().flatten().mean())

                                        # # plot prediction
                                        # if (plot_prediction == True) & (t > plot_start-1) & (t < plot_end):

                                        #     # visualize
                                        #     axs[1, t-plot_start+1].imshow(np.transpose(torch.Tensor(pred[0][:, :, :].detach().cpu().numpy()), (1, 2, 0)))
                                        #     axs[2, t-plot_start+1].imshow(torch.Tensor(E_seq[0][0, 4, :, :].detach().cpu().numpy()))

                                        #     # adjust axes
                                        #     axs[0, t-plot_start].axis('off')
                                        #     axs[1, t-plot_start].axis('off')
                                        #     axs[2, t-plot_start].axis('off')

                                # # adjust axes
                                # if plot_prediction:
                                #     axs[0, 0].axis('off')
                                #     axs[1, 0].axis('off')
                                #     axs[2, 0].axis('off')

                                #     axs[0, plot_end-plot_start].axis('off')
                                #     axs[1, plot_end-plot_start].axis('off')
                                #     axs[2, plot_end-plot_start].axis('off')
                                        
                                #     # save figure
                                #     plt.tight_layout()
                                #     plt.savefig(root + 'visualization/prediction/Kirubeswaran2023/Kirubeswaran2023_' + trial + '_frame_prediction_ ', dpi=600)

                        # save activations
                        for iL in range(n_layer):
                            if trained:
                                np.save(root_save + '/analyse_' + analyse + '/activations/' + dataset + '/' + test_set_current + '_' + train_set + '_' + trial + '_' + output_mode + str(iL+1) + '_actvs_trained_' + str(iInit+1), metrics_avg[iL])
                            else:
                                np.save(root_save + '/analyse_' + analyse + '/activations/' + dataset + '/' + test_set_current + '_' + train_set + '_'  + trial + '_' + output_mode + str(iL+1) + '_actvs_random_' + str(iInit+1), metrics_avg[iL])


if __name__ == '__main__':
    main()



