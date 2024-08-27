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
import prednet_Kirubeswaran2023
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

# define root stimuli
root        = '/home/amber/OneDrive/code/prednet_Brands2024/'
root_stim   = '/home/amber/Documents/prednet_Brands2024/data/stimuli/128_160/'

# test dataset
test_set        = ['set1', 'set2']

######################## ANALYSE ------- DATASET
# train_set = 'KITTI'
# train_set = 'WT_AMS'
# train_set = 'WT_VEN'
# train_set = 'WT_WL'

############################ ANALYSE ------- FPS
train_sets = ['WT_VEN_fps3', 'WT_VEN_fps6', 'WT_VEN_fps12']

########################### ANALYSE ------- LOSS
# train_sets = ['WT_VEN_Lnull', 'WT_VEN_Lall']

# select folder
if ('fps' in train_sets[0]):
    analyse = 'fps'
elif ('Lnull' in train_sets[0]) | ('Lall' in train_sets[0]):
    analyse = 'loss'
else:
    analyse = 'dataset'

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

    # number of random network initializations
    init                = 3

    # trial types
    # trials = np.array(['onepulse', 'twopulse_repeat', 'twopulse_nonrepeat_same'])
    trials = np.array(['onepulse'])

    # output mode
    output_modes = ['E']
    for output_mode in output_modes:
        assert output_mode in ['E', 'R', 'A', 'Ahat']

    # iterate over training dataset and retrieve activations
    for train_set in train_sets:

        # summarize
        print('trials: '.ljust(25), trials)
        print('output mode: '.ljust(25), output_modes)
        print('trained: '.ljust(25), trained)
        print('training set: '.ljust(25), train_set)

        # root for save
        if 'KITTI' in train_set:
            dir_save = 'KITTI'
        elif 'WT_AMS' in train_set:
            dir_save = 'WT_AMS'
        elif 'WT_VEN' in train_set:
            dir_save = 'WT_VEN'
        elif 'WT_WL' in train_set:
            dir_save = 'WT_WL'

        for iInit in range(init):

            # initiate model
            net = prednet_Kirubeswaran2023.PredNet(channels, device=device).to(device)
            net.eval()

            # load weights
            if train_set != 'random':
                initmodel = root + 'weights/Brands2024/' + analyse + '/' + train_set + '_' + str(iInit+1) + '.pth'
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
                                np.save('/home/amber/Documents/prednet_Brands2024/data/model/Kirubeswaran2023/datasets/analyse_' + analyse + '/' + dir_save + '/' + test_set_current + '_' + train_set + '_' + trial + '_' + output_mode + str(iL+1) + '_actvs_trained_' + str(iInit+1), metrics_avg[iL])
                            else:
                                np.save('/home/amber/Documents/prednet_Brands2024/data/model/Kirubeswaran2023/datasets/analyse_' + analyse + '/' + dir_save + '/' + test_set_current + '_' + train_set + '_'  + trial + '_' + output_mode + str(iL+1) + '_actvs_random_' + str(iInit+1), metrics_avg[iL])


if __name__ == '__main__':
    main()



