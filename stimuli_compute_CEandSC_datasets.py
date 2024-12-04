import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import yaml
import cv2
from sklearn.utils import resample
from lgnpy.lgnpy.CEandSC.lgn_statistics import lgn_statistics
import scipy
from sklearn.utils import resample

# select directory to save stimuli
root            = '/prednet_Brands2024_git/'
config_path     = '/prednet_Brands2024_git/lgnpy/lgnpy/CEandSC/default_config.yml'
data_save       = '/prednet_Brands2024_git/data/stimuli/img_statistics/'

# set dataset
# datasets = ['KITTI', 'WT_AMS', 'WT_WL']
# n_frames = [41396, 147377, 108115]

datasets = ['KITTI']
n_frames = [41396]

# datasets            = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']
# datasets_lbls       = ['KITTI', 'Amsterdam', 'Venice', 'Wild life']
# # color               = ['#DDCC77', '#117733', '#88CCEE', '#882255']

# sample
n_img = 2000

# compute
preload = False

if preload == False:

    # configure
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    print(config)

    # se threshold
    threshold_lgn = scipy.io.loadmat('/home/amber/OneDrive/code/prednet_Brands2024/lgnpy/ThresholdLGN.mat')['ThresholdLGN']

    # retrieve random sample
    idxs = np.zeros((len(datasets), n_img), dtype=int)
    for iD in range(len(datasets)):
        idxs[iD, :] = resample(np.arange(n_frames[iD]), replace=False, n_samples=n_img)

    # compute values
    for iD, dataset in enumerate(datasets):

        imgs = np.zeros((n_img))

        # initiate dataframe to store CE and SC values
        CEandSC_values = np.zeros((n_img, 2))

        # Define the path to the .mp4 video file
        file_path = '/home/amber/OneDrive/datasets/train/' + dataset + '.mp4'

        # import dataset
        cap = cv2.VideoCapture(file_path)

        # initiate dataframe to store CE and SC values
        CEandSC_values = np.zeros((n_img, 2))

        # Load all frames
        count = 0
        for f in range(n_frames[iD]):

            # read frame
            ret, frame = cap.read()

            # check in sample
            if int(f) in idxs[iD, :].tolist():

                # convert to rgb
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                
                # compute statistics
                ce, sc, _, _ = lgn_statistics(im=np.array(frame), file_name=str(count+1), config=config, force_recompute=True, cache=False, home_path=config_path, threshold_lgn=threshold_lgn)

                # save values
                CEandSC_values[count, 0] = ce[:, 0, 0].mean()
                CEandSC_values[count, 1] = sc[:, 0, 0].mean()

                # increment count
                count+=1
                
        cap.release()

        # save array
        np.save(data_save + dataset, CEandSC_values)


