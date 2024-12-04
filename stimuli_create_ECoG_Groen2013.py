import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.utils import resample
import os.path as path

# input shape for PredNet
input_shape = [128, 160, 3]

# select directory to save stimuli
root            = '/prednet_Brands2024_git'
root_stim       = '/datasets/Groen2013/'
root_vis        = '/prednet_Brands2024_git/visualization/stimuli/Groen2013/'
root_data       = '/prednet_Brands2024_git/data/stimuli/Groen2013/'

# plot stimuli
compute_onepulse                        = True
# compute_twopulse_repeat                 = True
# compute_twopulse_nonrepeat_same         = True
# compute_twopulse_nonrepeat_diff         = True

# number of images
img_n               = 1600
stim_duration       = 32

example_plot = random.randint(0, img_n-1)

# trials = np.array(['onepulse', 'twopulse_repeat', 'twopulse_nonrepeat_same'])
trials = np.array(['twopulse_repeat'])

################################################################ RESIZED
########################################################################

for iT, trial in enumerate(trials):

    # create stimulus for onepulse trials
    nt          = 45
    start       = 4
    tempCond    = [1, 2, 4, 8, 16, 32]

    # initiate dataframe
    # imgs_onepulse = np.ones((img_n, nt, input_shape[2], input_shape[0], input_shape[1]), dtype=np.float32)*0.5
    
    shape      = (img_n, nt, input_shape[2], input_shape[0], input_shape[1])
    imgs       = np.memmap(root_data + 'imgs_' + trial, dtype='float32', mode='w+', shape=shape)
    imgs[:]    = 0.5
    
    # Ensure data is written to disk
    imgs.flush()

    # compute one-pulse
    for iImg in range(img_n):

        # print progress
        print('Img ', iImg+1, '...')

        # import image
        if iImg < 9:
            img = Image.open(root_stim + 'im_000' + str(iImg+1) + '.jpg')
        elif iImg < 99:
            img = Image.open(root_stim + 'im_00' + str(iImg+1) + '.jpg')
        elif iImg < 999:
            img = Image.open(root_stim + 'im_0' + str(iImg+1) + '.jpg')
        else:
            img = Image.open(root_stim + 'im_' + str(iImg+1) + '.jpg')

        # resize
        img = img.resize((input_shape[1], input_shape[0]))
        img = np.array(img)

        # add to dataset
        if trial == 'onepulse':
            imgs[iImg, start:start+stim_duration, :, :, :] = np.transpose(img[:, :, :], (2, 0, 1))/255
        elif trial == 'twopulse_repeat':
            imgs[iImg, start:start+1, :, :, :] = np.transpose(img[:, :, :], (2, 0, 1))/255
            imgs[iImg, start+2:start+3, :, :, :] = np.transpose(img[:, :, :], (2, 0, 1))/255

        # Ensure data is written to disk
        imgs.flush()

        # visualize
        if iImg == example_plot:

            # initiate figure
            fig, axs = plt.subplots(1, nt, figsize=(20, 3), facecolor='white')

            # visualize
            for t in range(nt):
                
                # plot
                axs[t].imshow(np.transpose(imgs[example_plot, t, :, :, :], (1, 2, 0)))

                # adjust axes
                axs[t].axis('off')

            # save
            plt.savefig(root_vis + 'imgs_' + trial, dpi=300, bbox_inches='tight')
            plt.close()


# ############################################################TWOPULSE-REP
# ########################################################################
# duration_twopulse = 1

# if compute_twopulse_repeat:

#     # create stimulus for onepulse trials
#     nt          = 45
#     start       = 4
#     duration    = duration_twopulse
#     tempCond    = [1, 2, 4, 8, 16, 32]

#     # # initiate dataframe
#     imgs_twopulse_repeat = np.ones((len(tempCond), img_n, nt, input_shape[2], input_shape[0], input_shape[1]), dtype=np.float32)*0.5

#     # compute one-pulse
#     for iT in range(len(tempCond)): # duration condition
#         for iD in range(duration): # iterate over timepoints
#             imgs_twopulse_repeat[iT, :, start+iD, :, :, :] = np.transpose(imgs[cat_idx, :, :, :, :], (0, 3, 1, 2))/255
#             imgs_twopulse_repeat[iT, :, start+duration+tempCond[iT]+iD, :, :, :] = np.transpose(imgs[cat_idx, :, :, :, :], (0, 3, 1, 2))/255

#     # initiate figure
#     fig, axs = plt.subplots(len(tempCond), nt, figsize=(20, 3), facecolor='white')

#     # visualize
#     idx     = idx = random.randint(0, img_n-1)
#     for iT in range(len(tempCond)):
#         for t in range(nt):
            
#             # plot
#             axs[iT, t].imshow(np.transpose(imgs_twopulse_repeat[iT, idx, t, :, :, :], (1, 2, 0)))

#             # adjust axes
#             axs[iT, t].axis('off')

#     # save
#     plt.savefig(root_vis + 'imgs_twopulse_repeat', dpi=300, bbox_inches='tight')
#     plt.close()

#     # save data
#     plt.tight_layout()
#     np.save(root_stim + 'stimuli_twopulse_repeat', imgs_twopulse_repeat)

# ##################################################### TWOPULSE-NREP-SAME
# ########################################################################

# if compute_twopulse_nonrepeat_same:

#     # create stimulus for onepulse trials
#     nt          = 45
#     start       = 4
#     duration    = duration_twopulse
#     tempCond    = [1, 2, 4, 8, 16, 32]

#     # initiate dataframe
#     imgs_twopulse_nonrepeat_same = np.ones((len(tempCond), img_n, nt, input_shape[2], input_shape[0], input_shape[1]), dtype=np.float32)*0.5

#     # compute one-pulse
#     for iT in range(len(tempCond)): # duration condition

#         # select indices
#         same_indices = True
#         while same_indices:

#             # choose indices
#             idx_img1 = resample(range(img_n), replace=False, n_samples=img_n)
#             idx_img2 = resample(range(img_n), replace=False, n_samples=img_n)
#             # print(idx_img1)

#             # check if same cat. indices.
#             same_indices = False
#             for iImg in range(img_n):
#                 if idx_img1[iImg] == idx_img2[iImg]:
#                     same_indices = True

#         for iD in range(duration): # iterate over timepoints
#             imgs_twopulse_nonrepeat_same[iT, :, start+iD, :, :, :] = np.transpose(imgs[cat_idx, idx_img1, :, :, :], (0, 3, 1, 2))/255
#             imgs_twopulse_nonrepeat_same[iT, :, start+duration+tempCond[iT]+iD, :, :, :] = np.transpose(imgs[cat_idx, idx_img2, :, :, :], (0, 3, 1, 2))/255

#     # initiate figure
#     fig, axs = plt.subplots(len(tempCond), nt, figsize=(20, 3), facecolor='white')

#     # visualize
#     idx     = idx = random.randint(0, img_n-1)
#     for iT in range(len(tempCond)):
#         for t in range(nt):
            
#             # plot
#             axs[iT, t].imshow(np.transpose(imgs_twopulse_nonrepeat_same[iT, idx, t, :, :, :], (1, 2, 0)))

#             # adjust axes
#             axs[iT, t].axis('off')

#     # save
#     plt.savefig(root_vis + 'imgs_twopulse_nonrepeat_same', dpi=300, bbox_inches='tight')
#     plt.close()

#     # save data
#     plt.tight_layout()
#     np.save(root_stim + 'stimuli_twopulse_nonrepeat_same', imgs_twopulse_nonrepeat_same)

# ##################################################### TWOPULSE-NREP-DIFF
# ########################################################################

# if compute_twopulse_nonrepeat_diff:

#     # create stimulus for onepulse trials
#     nt          = 45
#     start       = 4
#     duration    = duration_twopulse
#     tempCond    = [1, 2, 4, 8, 16, 32]

#     # initiate dataframe
#     imgs_twopulse_nonrepeat_diff = np.ones((len(tempCond), img_n, nt, input_shape[2], input_shape[0], input_shape[1]), dtype=np.float32)*0.5

#     # compute one-pulse
#     for iT in range(len(tempCond)): # duration condition

#         # select random images
#         idx_img = resample(range(img_n), replace=False, n_samples=img_n)
#         cat_img2 = resample(cat_idx_other, replace=True, n_samples=img_n)

#         for iD in range(duration): # iterate over timepoints

#             # choose other classes
#             imgs_twopulse_nonrepeat_diff[iT, :, start+iD, :, :, :] = np.transpose(imgs[cat_idx, idx_img, :, :, :], (0, 3, 1, 2))/255
#             imgs_twopulse_nonrepeat_diff[iT, :, start+duration+tempCond[iT]+iD, :, :, :] = np.transpose(imgs[cat_img2, idx_img, :, :, :], (0, 3, 1, 2))/255

#     # initiate figure
#     fig, axs = plt.subplots(len(tempCond), nt, figsize=(20, 3), facecolor='white')

#     # visualize
#     idx     = idx = random.randint(0, img_n-1)
#     for iT in range(len(tempCond)):
#         for t in range(nt):
            
#             # plot
#             axs[iT, t].imshow(np.transpose(imgs_twopulse_nonrepeat_diff[iT, idx, t, :, :, :], (1, 2, 0)))

#             # adjust axes
#             axs[iT, t].axis('off')

#     # save
#     plt.savefig(root_vis + 'imgs_twopulse_nonrepeat_diff', dpi=300, bbox_inches='tight')
#     plt.close()

#     # save data
#     plt.tight_layout()
#     np.save(root_stim + 'stimuli_twopulse_nonrepeat_diff', imgs_twopulse_nonrepeat_diff)