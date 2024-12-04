import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.utils import resample

# input shape for PredNet
input_shape = [128, 160, 3]

# choose dataset
dataset         = 'set1'

# select directory to save stimuli
root = '/prednet_Brands2024_git/'
if input_shape[0] == 120:
    root_stim   = '/prednet_Brands2024_git/data/stimuli/120_160/' + dataset + '/'
    root_vis    = root + 'visualization/stimuli/120_160/' + dataset + '/'
elif input_shape[0] == 128:
    root_stim   = '/prednet_Brands2024_git/data/stimuli/128_160/' + dataset + '/'
    root_vis    = root + '/visualization/stimuli/128_160/' + dataset + '/'

# categories
cats = ['bodies', 'buildings', 'faces', 'objects', 'scenes', 'scrambled']

# plot stimuli
compute_resized                         = False
compute_onepulse                        = False
compute_twopulse_repeat                 = True
compute_twopulse_nonrepeat_same         = False
compute_twopulse_nonrepeat_diff         = False

# img info
img_n           = 24
cat_idx         = 4
cat_idx_other   = [0, 1, 2, 3, 4, 5]
cat_idx_other.remove(cat_idx)
print(cat_idx_other)

################################################################ RESIZED
########################################################################

if compute_resized:
    
    # initiate dataframes
    imgs = np.zeros((len(cats), img_n, input_shape[0], input_shape[1], input_shape[2]), dtype=int)

    # create images
    for iC in range(len(cats)):
        for iImg in range(img_n):

            # import from set1
            img1 = Image.open(root + 'stimuli/' + dataset + '/img' + str(iC*img_n+iImg+1) + '.jpg')
            imgs[iC, iImg, :, :, :] = img1.resize((input_shape[1], input_shape[0]))

    # initiate figure
    fig, axs = plt.subplots(1, len(cats), figsize=(20, 6), facecolor='white')

    for iC, cat in enumerate(cats):

        # select random integer
        idx = random.randint(0, img_n-1)

        # plot image
        axs[iC].imshow(imgs[iC, idx, :, :, :])

    # save
    plt.savefig(root_vis + 'all_cats', dpi=300)
    plt.close()

    # save data
    np.save(root_stim + 'stimuli_resized', imgs)

else:

    # load images
    imgs = np.load(root_stim + 'stimuli_resized.npy')
    print(imgs.size)

############################################################### ONEPULSE
########################################################################

if compute_onepulse:

    # create stimulus for onepulse trials
    nt          = 45
    start       = 4
    tempCond    = [1, 2, 4, 8, 16, 32]

    # initiate dataframe
    imgs_onepulse = np.ones((len(tempCond), img_n, nt, input_shape[2], input_shape[0], input_shape[1]), dtype=np.float32)*0.5

    # compute one-pulse
    for iT in range(len(tempCond)): # duration condition
        for iD in range(tempCond[iT]): # iterate over timepoints
            imgs_onepulse[iT, :, start+iD, :, :, :] = np.transpose(imgs[cat_idx, :, :, :, :], (0, 3, 1, 2))/255

    # initiate figure
    fig, axs = plt.subplots(len(tempCond), nt, figsize=(20, 3), facecolor='white')

    # visualize
    idx    = random.randint(0, img_n-1)
    for iT in range(len(tempCond)):
        for t in range(nt):
            
            # plot
            axs[iT, t].imshow(np.transpose(imgs_onepulse[iT, idx, t, :, :, :], (1, 2, 0)))

            # adjust axes
            axs[iT, t].axis('off')

    # save
    plt.savefig(root_vis + 'imgs_onepulse', dpi=300, bbox_inches='tight')
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
    np.save(root_stim + '/stimuli_onepulse', imgs_onepulse)

############################################################TWOPULSE-REP
########################################################################
duration_twopulse = 8

if compute_twopulse_repeat:

    # create stimulus for onepulse trials
    nt          = 55
    start       = 4
    duration    = duration_twopulse
    tempCond    = [1, 2, 4, 8, 16, 32]

    # # initiate dataframe
    imgs_twopulse_repeat = np.ones((len(tempCond), img_n, nt, input_shape[2], input_shape[0], input_shape[1]), dtype=np.float32)*0.5

    # compute one-pulse
    for iT in range(len(tempCond)): # duration condition
        for iD in range(duration): # iterate over timepoints
            imgs_twopulse_repeat[iT, :, start+iD, :, :, :] = np.transpose(imgs[cat_idx, :, :, :, :], (0, 3, 1, 2))/255
            imgs_twopulse_repeat[iT, :, start+duration+tempCond[iT]+iD, :, :, :] = np.transpose(imgs[cat_idx, :, :, :, :], (0, 3, 1, 2))/255

    # initiate figure
    fig, axs = plt.subplots(len(tempCond), nt, figsize=(20, 3), facecolor='white')

    # visualize
    idx     = idx = random.randint(0, img_n-1)
    for iT in range(len(tempCond)):
        for t in range(nt):
            
            # plot
            axs[iT, t].imshow(np.transpose(imgs_twopulse_repeat[iT, idx, t, :, :, :], (1, 2, 0)))

            # adjust axes
            axs[iT, t].axis('off')

    # save
    plt.savefig(root_vis + 'imgs_twopulse_repeat_8_duration', dpi=300, bbox_inches='tight')
    plt.close()

    # save data
    plt.tight_layout()
    np.save(root_stim + 'stimuli_twopulse_repeat_8_duration', imgs_twopulse_repeat)

##################################################### TWOPULSE-NREP-SAME
########################################################################

if compute_twopulse_nonrepeat_same:

    # create stimulus for onepulse trials
    nt          = 55
    start       = 4
    duration    = duration_twopulse
    tempCond    = [1, 2, 4, 8, 16, 32]

    # initiate dataframe
    imgs_twopulse_nonrepeat_same = np.ones((len(tempCond), img_n, nt, input_shape[2], input_shape[0], input_shape[1]), dtype=np.float32)*0.5

    # compute one-pulse
    for iT in range(len(tempCond)): # duration condition

        # select indices
        same_indices = True
        while same_indices:

            # choose indices
            idx_img1 = resample(range(img_n), replace=False, n_samples=img_n)
            idx_img2 = resample(range(img_n), replace=False, n_samples=img_n)
            # print(idx_img1)

            # check if same cat. indices.
            same_indices = False
            for iImg in range(img_n):
                if idx_img1[iImg] == idx_img2[iImg]:
                    same_indices = True

        for iD in range(duration): # iterate over timepoints
            imgs_twopulse_nonrepeat_same[iT, :, start+iD, :, :, :] = np.transpose(imgs[cat_idx, idx_img1, :, :, :], (0, 3, 1, 2))/255
            imgs_twopulse_nonrepeat_same[iT, :, start+duration+tempCond[iT]+iD, :, :, :] = np.transpose(imgs[cat_idx, idx_img2, :, :, :], (0, 3, 1, 2))/255

    # initiate figure
    fig, axs = plt.subplots(len(tempCond), nt, figsize=(20, 3), facecolor='white')

    # visualize
    idx     = idx = random.randint(0, img_n-1)
    for iT in range(len(tempCond)):
        for t in range(nt):
            
            # plot
            axs[iT, t].imshow(np.transpose(imgs_twopulse_nonrepeat_same[iT, idx, t, :, :, :], (1, 2, 0)))

            # adjust axes
            axs[iT, t].axis('off')

    # save
    plt.savefig(root_vis + 'imgs_twopulse_nonrepeat_same_8_duration', dpi=300, bbox_inches='tight')
    plt.close()

    # save data
    plt.tight_layout()
    np.save(root_stim + 'stimuli_twopulse_nonrepeat_same_8_duration', imgs_twopulse_nonrepeat_same)

##################################################### TWOPULSE-NREP-DIFF
########################################################################

if compute_twopulse_nonrepeat_diff:

    # create stimulus for onepulse trials
    nt          = 55
    start       = 4
    duration    = duration_twopulse
    tempCond    = [1, 2, 4, 8, 16, 32]

    # initiate dataframe
    imgs_twopulse_nonrepeat_diff = np.ones((len(tempCond), img_n, nt, input_shape[2], input_shape[0], input_shape[1]), dtype=np.float32)*0.5

    # compute one-pulse
    for iT in range(len(tempCond)): # duration condition

        # select random images
        idx_img = resample(range(img_n), replace=False, n_samples=img_n)
        cat_img2 = resample(cat_idx_other, replace=True, n_samples=img_n)

        for iD in range(duration): # iterate over timepoints

            # choose other classes
            imgs_twopulse_nonrepeat_diff[iT, :, start+iD, :, :, :] = np.transpose(imgs[cat_idx, idx_img, :, :, :], (0, 3, 1, 2))/255
            imgs_twopulse_nonrepeat_diff[iT, :, start+duration+tempCond[iT]+iD, :, :, :] = np.transpose(imgs[cat_img2, idx_img, :, :, :], (0, 3, 1, 2))/255

    # initiate figure
    fig, axs = plt.subplots(len(tempCond), nt, figsize=(20, 3), facecolor='white')

    # visualize
    idx     = idx = random.randint(0, img_n-1)
    for iT in range(len(tempCond)):
        for t in range(nt):
            
            # plot
            axs[iT, t].imshow(np.transpose(imgs_twopulse_nonrepeat_diff[iT, idx, t, :, :, :], (1, 2, 0)))

            # adjust axes
            axs[iT, t].axis('off')

    # save
    plt.savefig(root_vis + 'imgs_twopulse_nonrepeat_diff_8_duration', dpi=300, bbox_inches='tight')
    plt.close()

    # save data
    plt.tight_layout()
    np.save(root_stim + 'stimuli_twopulse_nonrepeat_diff_8_duration', imgs_twopulse_nonrepeat_diff)