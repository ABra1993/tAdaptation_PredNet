import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageCms
import argparse
import datetime
from tqdm import tqdm
from distutils.util import strtobool
import time
from datetime import datetime

# get the start time
st = time.time()

# torch utilities
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

# other scripts
from model_train_Kirubeswaran2023_utils_corr_wise import CorrWise

from torch.utils.data import Dataset, DataLoader
from model_train_Kirubeswaran2023_utils_dataloader import VideoDataset
from prednet_Kirubeswaran2023 import *

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define root stimuli
root        = '/home/amber/OneDrive/code/prednet_Brands2024/'
root_stim   = '/home/amber/Documents/prednet_Brands2024/data/stimuli/128_160/'

# training dataset (model init.)
# train_set = 'KITTI'
# train_set = 'WT_AMS'
# train_set = 'WT_VEN'
# train_set = 'WT_WL'

train_sets = ['WT_VEN']
# train_sets = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']

# analyse
analyse = 'fps'
if analyse == 'fps':
    manip = ['fps3', 'fps12']
elif analyse == 'loss':
    manip = ['Lnull', 'Lall']

# test dataset
# test_set        = ['set1', 'set2']

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
    sequence_length     = 10
    samples             = 10

    # input size
    w = 160
    h = 128
    c = 3

    # number of random network initializations
    init                = 1

    # network settings
    channels            = [3, 48, 96, 192]

    # iterate over training dataset and retrieve activations
    for train_set in train_sets:

        # Define the path to the .mp4 video file
        file_path = '/home/amber/OneDrive/datasets/train/' + train_set + '.mp4'

        # Create a VideoDataset instance
        video_dataset = VideoDataset(False, file_path, sequence_length, w, h, c)

        # Calculate the number of batches
        num_batches = len(video_dataset)

        # determine if the last batch should be removed
        remove_last_batch = (video_dataset.total_frames % sequence_length != 0)
        print('Remove last batch: ', remove_last_batch)

        # create a DataLoader without shuffling
        if remove_last_batch:
            num_batches -= 1
        data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False)
        print(len(data_loader))

        for iInit in range(init):

            # load weights Lall
            net_Lnull = PredNet(channels, device=device).to(device)
            initmodel_Lnull = root + 'weights/Brands2024/' + analyse + '/' + train_set + '_' + manip[0] + '_' + str(iInit+1) + '.pth'
            net_Lnull.load_state_dict(torch.load(initmodel_Lnull))
            net_Lnull.eval()

            # load weights Lnull
            net_Lall = PredNet(channels, device=device).to(device)
            initmodel_Lall = root + 'weights/Brands2024/' + analyse + '/' + train_set + '_' + manip[1] + '_' +  str(iInit+1) + '.pth'
            net_Lall.load_state_dict(torch.load(initmodel_Lall))
            net_Lall.eval()

            for sample_idx in range(samples):

                # choose batch
                batch = next(iter(data_loader))
                print(batch.shape)

                # initiate figure
                fig, axs = plt.subplots(4, sequence_length, figsize=(10, 4))

                # # select current data
                # current_data = batch[:, :t+2, :, :, :]
                # print('Data: ', current_data.shape)

                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True):
                        
                        # predict
                        pred_Lnull, _, _, _, _ = net_Lnull(batch.to(device))

                        # predict
                        pred_Lall, _, _, _, _ = net_Lall(batch.to(device))

                # plot outputs
                for t in range(sequence_length): # last index 44
                
                    # plot ground truth
                    axs[0, t].imshow(batch[0, t, :, :, :].permute(1, 2, 0))
                    axs[0, t].axis('off')

                    if t == 0:
                        continue

                    # plot prediction Lnull
                    axs[1, t].imshow(np.transpose(torch.Tensor(pred_Lnull[t-1][:, :, :].squeeze().detach().cpu().numpy()), (1, 2, 0)))
                    axs[1, t].axis('off')

                    # plot prediction Lall
                    axs[2, t].imshow(np.transpose(torch.Tensor(pred_Lall[t-1][:, :, :].squeeze().detach().cpu().numpy()), (1, 2, 0)))
                    axs[2, t].axis('off')

                    # plot difference
                    pred_diff = pred_Lnull[t-1] - pred_Lall[t-1]
                    # print(torch.max(pred_Lall[t-1].detach().cpu()))
                    # print(torch.max(pred_Lnull[t-1].detach().cpu()))

                    pred_diff = torch.clamp(pred_diff, 0, 1)
                    # axs[3, t].imshow(np.transpose(torch.Tensor(pred_diff.squeeze().detach().cpu().numpy()), (1, 2, 0)), vmin=0, vmax=1)
                    axs[3, t].imshow(np.transpose(torch.Tensor(pred_diff.squeeze().detach().cpu().numpy()), (1, 2, 0)), vmin=0, vmax=1)
                    axs[3, t].axis('off')

                # plot first and last timestep
                axs[1, 0].imshow(torch.ones_like(batch[0, -1, :, :, :].permute(1, 2, 0))*0.5)
                axs[1, 0].axis('off')

                axs[2, 0].imshow(torch.ones_like(batch[0, -1, :, :, :].permute(1, 2, 0))*0.5)
                axs[2, 0].axis('off')

                axs[3, 0].imshow(torch.ones_like(batch[0, -1, :, :, :].permute(1, 2, 0))*0.5)
                axs[3, 0].axis('off')

                # save figure
                # plt.tight_layout()
                plt.savefig('visualization/model/Kirubeswaran2023/datasets/feature_maps_' + analyse + '/' + train_set + str(sample_idx+1), bbox_inches='tight')
                plt.close()


if __name__ == '__main__':
    main()



