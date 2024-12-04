import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import yaml
import cv2
from sklearn.utils import resample
# from lgnpy.lgnpy.CEandSC.lgn_statistics import lgn_statistics
import scipy
from sklearn.utils import resample
import seaborn as sns

# select directory to save stimuli
root            = '/home/amber/OneDrive/code/prednet_Brands2024/'
data_save       = '/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/'

# datasets
datasets            = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']
datasets_lbls       = ['KITTI', 'Amsterdam', 'Venice', 'Wild life']
color               = ['#DDCC77', '#117733', '#88CCEE', '#882255']

# sample
n_frames = 25000

# compute
preload = False

if preload == False:

    # compute values
    for iD, dataset in enumerate(datasets):

        print(dataset)

        imgs = np.zeros((n_frames))

        # initiate dataframe to store CE and SC values
        CEandSC_values = np.zeros((n_frames, 2))

        # Define the path to the .mp4 video file
        # file_path = '/home/amber/OneDrive/datasets/train/' + dataset + '.mp4'
        file_path = '/home/amber/Documents/organize_stimuli/datasets/train/' + dataset + '.mp4'

        # import dataset
        cap = cv2.VideoCapture(file_path)

        # Read the first frame
        ret, prev_frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame.")

        # Convert the first frame to grayscale
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # compute cross correlation
        correlations = []
        count = 0
        while count < n_frames:

            print(count)

            # Read the next frame
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Convert the current frame to grayscale
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Flatten the frames
            prev_flat = prev_frame_gray.flatten()
            current_flat = current_frame_gray.flatten()
            
            # Check standard deviations to avoid division by zero
            if np.std(prev_flat) == 0 or np.std(current_flat) == 0:
                correlation = 0  # Assign a correlation of 0 if one of the frames is uniform
            else:
                # Calculate the correlation between the current frame and the previous frame
                correlation = np.corrcoef(prev_flat, current_flat)[0, 1]
                correlations.append(correlation)

            # Update the previous frame
            prev_frame_gray = current_frame_gray

            # increment count
            count+=1
        
        cap.release()

        # save array
        np.save(data_save + 'crossCorr_' + dataset, np.mean(correlations))
        np.save(data_save + 'crossCorr_' + dataset + '_sample', correlations)


