import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.utils import resample
import cv2
import torch

# input shape for PredNet
input_shape = [128, 160, 3]
w = 160
h = 128
c = 3

# choose dataset
# dataset         = 'KITTI'
# dataset         = 'WT_AMS'
# dataset         = 'WT_VEN'
dataset         = 'WT_WL'

# select directory to save stimuli
root        = '/prednet_Brands2024_git/'
root_stim   = '/prednet_Brands2024_git/data/stimuli/128_160/'
root_vis    = '/prednet_Brands2024_git/visualization/stimuli/128_160/'

# open video
video_path = '/home/amber/OneDrive/datasets/train/' + dataset + '.mp4'
cap = cv2.VideoCapture(video_path)
sequence_length = 10

# Load all frames
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (w, h))  # Resize frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = frame / 255.0  # Normalize
    frame = torch.Tensor(frame).permute(2, 0, 1)  # Reshape to (C, H, W)
    frames.append(frame)
cap.release()

total_frames = len(frames)
print('Total number of frames:', total_frames)

# categories
cats = ['bodies', 'buildings', 'faces', 'objects', 'scenes', 'scrambled']

# plot stimuli
compute_resized                         = True
compute_onepulse                        = True
compute_twopulse_repeat                 = True
compute_twopulse_nonrepeat_same         = True
compute_twopulse_nonrepeat_diff         = True

############################################################### ONEPULSE
########################################################################

if compute_onepulse:

    # create stimulus for onepulse trials
    n_img       = 48
    nt          = 45
    start       = 4
    tempCond    = [1, 2, 4, 8, 16, 32]

    # initiate dataframe
    imgs_onepulse = np.ones((len(tempCond), n_img, nt, input_shape[2], input_shape[0], input_shape[1]), dtype=np.float32)*0.5

    # visualize
    for iImg in range(n_img):

        # select random frame
        start_idx = random.randint(0, total_frames - nt)

        # compute one-pulse
        for iT in range(len(tempCond)): # duration condition
            for iD in range(tempCond[iT]): # iterate over timepoints
                imgs_onepulse[iT, iImg, start+iD, :, :, :] = frames[start_idx]

        if iImg == 0:

            # initiate figure
            fig, axs = plt.subplots(len(tempCond), nt, figsize=(10, 2), facecolor='white')

            # plot
            for iT in range(len(tempCond)):
                for t in range(nt):
                    
                    # plot
                    axs[iT, t].imshow(np.transpose(imgs_onepulse[iT, iImg, t, :, :, :], (1, 2, 0)))

                    # adjust axes
                    axs[iT, t].axis('off')

            # save
            # plt.tight_layout()
            plt.savefig(root_vis + 'datasets/imgs_onepulse_' + dataset, dpi=300, bbox_inches='tight')
            plt.close()

    # # plot stream for pytorch implementation
    # with open('/home/amber/OneDrive/code/prednet_Kirubeswaran2023/prednet_in_pytorch/imgs/test.txt', 'w') as f:
    #     for t in range(nt):
            
    #         # file path
    #         path = '/home/amber/OneDrive/code/prednet_Kirubeswaran2023/prednet_in_pytorch/imgs/' + str(t+1) + '.jpg'
            
    #         # visualize
    #         fig = plt.figure()
    #         plt.imshow(np.transpose(imgs_onepulse[-1, idx, t, :, :, :], (1, 2, 0)))
    #         plt.savefig(path)
    #         plt.close()

    #         f.writelines(path + '\n')

    # save data
    plt.tight_layout()
    np.save(root_stim + 'datasets/stimuli_onepulse_' + dataset, imgs_onepulse)
