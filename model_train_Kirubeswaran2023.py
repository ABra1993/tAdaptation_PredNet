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
# from model_train_Kirubeswaran2023_utils_dataset import ImageListDataset
from model_train_Kirubeswaran2023_utils_corr_wise import CorrWise

from torch.utils.data import Dataset, DataLoader
from model_train_Kirubeswaran2023_utils_dataloader import VideoDataset
from prednet_Kirubeswaran2023 import *

parser = argparse.ArgumentParser(description='PredNet')
parser.add_argument('--images', '-i', default='data/train_list.txt', help='Path to image list file')
parser.add_argument('--sequences', '-seq', default='', help='Path to sequence list file')
parser.add_argument('--device', '-d', default="", type=str,
                    help='Computational device')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of sequence and image files')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of target images. width,height (pixels)')
parser.add_argument('--channels', '-c', default='3,48,96,192',
                    help='Number of channels on each layers')
parser.add_argument('--offset', '-o', default='0,0',
                    help='Center offset of clipping input image (pixels)')
parser.add_argument('--input_len', '-l', default=20, type=int,
                    help='Input frame length fo extended prediction on test (frames)')
parser.add_argument('--ext', '-e', default=10, type=int,
                    help='Extended prediction on test (frames)')
parser.add_argument('--bprop', default=20, type=int,
                    help='Back propagation length (frames)')
parser.add_argument('--save', default=10000, type=int,
                    help='Period of save model and state (frames)')
parser.add_argument('--period', default=1000000, type=int,
                    help='Period of training (frames)')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--saveimg', dest='saveimg', action='store_true')
parser.add_argument('--useamp', dest='useamp', action='store_true', help='Flag for using AMP')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--lr_rate', default=1.0, type=float,
                    help='Reduction rate for Step lr scheduler')
parser.add_argument('--min_lr', default=0.0001, type=float,
                    help='Lower bound learning rate for Step lr scheduler')
parser.add_argument('--batchsize', default=1, type=int, help='Input batch size')
parser.add_argument('--shuffle', default=False, type=strtobool, help=' True is enable to sampl data randomly (default: False)')
parser.add_argument('--num_workers', default=0, type=int, help='Num. of dataloader process. (default: num of cpu cores')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='True is enable to log for Tensorboard')
parser.add_argument('--up_down_up', action='store_true', help='True is enable to cycle up-down-up in order')
parser.add_argument('--color_space', default='RGB', type=str, help='Image color space(RGB, HSV, LAB, CMYK, YcbCr) - the dimension of this color space and 1st channel must be same.')
parser.add_argument('--loss', type=str, default='mse', help='Loss name for training. Please select loss from "mse", "corr_wise", and "ensemble" (default: mse).')
parser.set_defaults(test=False)
args = parser.parse_args()


parser = argparse.ArgumentParser(description='PredNet')

# network inputs
parser.add_argument('--images', '-i', default='data/train_list.txt', help='Path to image list file')
parser.add_argument('--sequences', '-seq', default='', help='Path to sequence list file')

# device and root directory
parser.add_argument('--device', '-d', default="", type=str,
                    help='Computational device')

parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of sequence and image files')

# init model (e.g. finetuning)?
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')

# input size and number of channels
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of target images. width,height (pixels)')
parser.add_argument('--channels', '-c', default='3,48,96,192',
                    help='Number of channels on each layers')
parser.add_argument('--input_len', '-l', default=20, type=int,
                    help='Input frame length fo extended prediction on test (frames)')

# training settings
parser.add_argument('--bprop', default=20, type=int,
                    help='Back propagation length (frames)')

# saving frequency
parser.add_argument('--save', default=10000, type=int,
                    help='Period of save model and state (frames)')

# training period
parser.add_argument('--period', default=800000, type=int,
                    help='Period of training (frames)')

# leraning rate
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--lr_rate', default=1.0, type=float,
                    help='Reduction rate for Step lr scheduler')
parser.add_argument('--min_lr', default=0.0001, type=float,
                    help='Lower bound learning rate for Step lr scheduler')

# other
parser.add_argument('--useamp', dest='useamp', action='store_true', help='Flag for using AMP')
parser.add_argument('--batchsize', default=1, type=int, help='Input batch size')
parser.add_argument('--shuffle', default=False, type=strtobool, help=' True is enable to sampl data randomly (default: False)')
parser.add_argument('--num_workers', default=0, type=int, help='Num. of dataloader process. (default: num of cpu cores')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='True is enable to log for Tensorboard')
parser.add_argument('--up_down_up', action='store_true', help='True is enable to cycle up-down-up in order')
parser.add_argument('--color_space', default='RGB', type=str, help='Image color space(RGB, HSV, LAB, CMYK, YcbCr) - the dimension of this color space and 1st channel must be same.')
parser.add_argument('--loss', type=str, default='mse', help='Loss name for training. Please select loss from "mse", "corr_wise", and "ensemble" (default: mse).')
parser.add_argument('--saveimg', dest='saveimg', action='store_true')

args = parser.parse_args()

def train(device=torch.device("cpu")):

    # set the seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # training settings
    init                    = 3
    nb_epoch                = 150
    batch_size              = 4
    samples_per_epoch       = 500

    # init                    = 1
    # nb_epoch                = 1
    # batch_size              = 1
    # samples_per_epoch       = 1

    # Define the sequence length
    sequence_length = 10

    # print GPU support
    print(device)

    ######################## ANALYSE ------- DATASET

    # dataset = 'KITTI'
    # dataset = 'WT_AMS'
    # dataset = 'WT_VEN'
    # dataset = 'WT_WL'

    ############################ ANALYSE ------- FPS

    # dataset = 'KITTI_fps3'
    # dataset = 'KITTI_fps6'
    # dataset = 'KITTI_fps12'

    # dataset = 'WT_VEN_fps3'
    # dataset = 'WT_VEN_fps6'
    # dataset = 'WT_VEN_fps12'

    # dataset = 'WT_AMS_fps3'
    # dataset = 'WT_AMS_fps6'
    # dataset = 'WT_AMS_fps12'

    # dataset = 'WT_WL_fps3'
    # dataset = 'WT_WL_fps6'
    # dataset = 'WT_WL_fps12'

    ########################### ANALYSE ------- loss

    # dataset = 'KITTI_Lnull'
    # dataset = 'KITTI_Lall'

    # dataset = 'WT_VEN_Lnull'
    # dataset = 'WT_VEN_Lall'

    # dataset = 'WT_AMS_Lnull'
    # dataset = 'WT_AMS_Lall'

    dataset = 'WT_WL_Lnull'
    dataset = 'WT_WL_Lall'

    # select folder
    if ('fps' in dataset):
        analyse = 'fps'
    elif ('Lnull' in dataset) | ('Lall' in dataset):
        analyse = 'loss'
    else:
        analyse = 'dataset'

    # dataset network is trained on
    if dataset == 'FPSI':
        w = 160
        h = 120
        c = 3
    elif ('KITTI' in dataset) | ('WT' in dataset):
        w = 160
        h = 128
        c = 3

    # define the path to the .mp4 video file
    if ('Lnull' in dataset) | ('Lall' in dataset):
        if 'KITTI' in dataset:
            path = 'KITTI'
        elif 'WT_AMS' in dataset:
            path = 'WT_AMS'
        elif 'WT_VEN' in dataset:
            path = 'WT_VEN'
        elif 'WT_WL' in dataset:
            path = 'WT_WL'
        file_path = '/home/amber/OneDrive/datasets/train/' + path + '.mp4'
    else:
        file_path = '/home/amber/OneDrive/datasets/train/' + dataset + '.mp4'

    print(file_path)
    static = False

    # create a VideoDataset instance
    video_dataset = VideoDataset(static, file_path, sequence_length, w, h, c)
    
    # set weight losses
    if 'Lall' in dataset:
        loss_weight         = [1, 0.1, 0.1, 0.1]
    else:
        loss_weight         = [1, 0, 0, 0]

    # print summary
    print(30*'--')
    print('Trained on: '.ljust(30), dataset)
    print('Trained on static images: '.ljust(30), static)
    print('Loss: '.ljust(30), args.loss)
    print('AMP: '.ljust(30), args.useamp)
    print(30*'--')

    # train
    for iInit in range(init):

        # open log file
        logf = open('/home/amber/OneDrive/code/prednet_Brands2024/losses/' + analyse + '/' + dataset + '_' + str(iInit+1) + '_log_t.txt', 'w')

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

        # initiate Tensorboard
        writer = SummaryWriter() # if args.tensorboard else None

        # initiate PredNet
        net = PredNet(args.channels, loss_weight=loss_weight,
                            round_mode="up_donw_up" if args.up_down_up else "down_up_down",
                            device=device).to(device)
        
        # print loss weight
        print('Loss weight: ', net.loss_weight)

        # initiate loss
        base_loss = nn.L1Loss()
        loss = CorrWise(base_loss, flow_method="FBFlow", return_warped=False, reduction_clip=False, flow_cycle_loss=True, scale_clip=True, device=device)
        
        # optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        # AMP
        scaler = torch.cuda.amp.GradScaler(enabled=args.useamp) # AMP
        
        # schedule learning rate
        lr_maker = lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=args.lr_rate)

        # iterate
        step = 0
        for epoch in range(nb_epoch):
            print(f"Epoch {epoch + 1}/{nb_epoch}")
            for sample_idx in range(samples_per_epoch):

                batch = next(iter(data_loader))
                print(batch.shape)

                # # print first sequence
                # if (epoch == 0) & (sample_idx == 0):

                #     _, axs = plt.subplots(1, 10)
                    
                #     for tmp in range(sequence_length):
                #         axs[tmp].imshow(batch[0, tmp, :, :, :].permute(1, 2, 0))
                #         axs[tmp].axis('off')

                #     # save figure
                #     plt.tight_layout()
                #     plt.savefig('visualization/stimuli/datasets/static1_' + dataset + str(iInit+1), bbox_inches='tight')
                #     plt.close()

                #     _, axs = plt.subplots(1, 10)
                    
                #     for tmp in range(sequence_length):
                #         axs[tmp].imshow(batch[1, tmp, :, :, :].permute(1, 2, 0))
                #         axs[tmp].axis('off')

                #     # save figure
                #     plt.tight_layout()
                #     plt.savefig('visualization/stimuli/datasets/static2_' + dataset + str(iInit+1), bbox_inches='tight')
                #     plt.close()

                # sweep through network
                with torch.cuda.amp.autocast(enabled=args.useamp):

                    # forward
                    data = batch.to(device)
                    pred, errors, errors_per_layer, _, _  = net(data)

                    # compue loss
                    if args.loss == 'corr_wise':
                        mean_error = loss(pred, data[:, -1])
                    elif args.loss == 'ensemble':
                        corr_wise_error = loss(pred, data[:, -1])
                        mean_error = corr_wise_error + errors.mean()
                    elif args.loss == 'mse':
                        if ('Lnull' in dataset) | ('Lall' in dataset):
                            mean_error = torch.mean(loss_weight[0] * errors_per_layer[0, :])\
                                            + torch.mean(loss_weight[1] * errors_per_layer[1, :])\
                                                + torch.mean(loss_weight[2] * errors_per_layer[2, :])\
                                                    + torch.mean(loss_weight[3] * errors_per_layer[3, :])
                        else:
                            mean_error = errors.mean()
                
                # update network
                optimizer.zero_grad()
                scaler.scale(mean_error).backward() # speed up training
                scaler.step(optimizer)
                scaler.update()
                if lr_maker.get_lr()[0] > args.min_lr:
                    lr_maker.step()
                else:
                    lr_maker.optimizer.param_groups[0]['lr'] = args.min_lr
                
                # write loss to tensorboard
                logf.write(str(mean_error.detach().cpu().numpy()) + '\n')
                if writer is not None:
                    writer.add_scalar("loss", mean_error.detach().cpu().numpy(), step)

                # # get the end time
                et = time.time()

                # print progress
                c = datetime.now()
                current_time = c.strftime('%H:%M:%S')
                print(current_time, ', HH: ', str(np.round((et-st)/60/60, 5)).ljust(10), ' ----- init ', int(iInit+1), '/', str(init).ljust(10), ' ----- epoch ', epoch+1, '/', str(nb_epoch).ljust(10), ' ----- sample ', sample_idx+1, '/', str(samples_per_epoch).ljust(10), ' ----- loss: ', mean_error.detach().cpu().numpy())

            # save model
            print("Save the model")
            torch.save(net.state_dict(), '/home/amber/OneDrive/code/prednet_Brands2024/weights/Brands2024/' + analyse + '/' + dataset + '_' + str(iInit+1) + '.pth')

        # print summary
        print(30*'--')
        print('Trained on: '.ljust(30), dataset)
        print('Trained on static images: '.ljust(30), static)
        print('Loss: '.ljust(30), args.loss)
        print('AMP: '.ljust(30), args.useamp)
        print(30*'--')


if __name__ == '__main__':
    args.size = args.size.split(',')
    for i in range(len(args.size)):
        args.size[i] = int(args.size[i])
    args.channels = args.channels.split(',')
    for i in range(len(args.channels)):
        args.channels[i] = int(args.channels[i])

    # set device
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == '' else args.device

    # train
    train(device)
