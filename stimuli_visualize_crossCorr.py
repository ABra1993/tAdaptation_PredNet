import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# select directory to save stimuli
root            = '/home/amber/OneDrive/code/prednet_Brands2024/'
data_save       = '/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/'

# datasets
datasets            = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']
datasets_lbls       = ['KITTI', 'Amsterdam', 'Venice', 'Wild life']
color               = ['#DDCC77', '#117733', '#88CCEE', '#882255']


# initiate figure
fig = plt.figure()
axs = plt.gca()
sns.despine(offset=10)

# fontsizes 
fontsize_title          = 20
fontsize_legend         = 12
fontsize_label          = 18
fontsize_tick           = 15

# visualize
offset_iD = [0, 3, 6, 9]
for iD, dataset in enumerate(datasets):

        print('Current dataset: ', dataset)

        # select data
        data_current = np.load(data_save + 'crossCorr_' + dataset + '.npy')

        # compute spread
        data_mean   = np.mean(data_current)

        # visualize
        axs.bar(offset_iD[iD], data_mean, color=color[iD])

# adjust axes
axs.tick_params(axis='both', labelsize=fontsize_tick)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# axs[1].set_xlabel('Dataset', fontsize=fontsize_label)
axs.set_ylabel(r'Avg. cross correlation', fontsize=fontsize_label)
axs.set_xticks(offset_iD)
axs.set_xticklabels(datasets_lbls, rotation=30, ha='right')

# save figure
plt.tight_layout()
# plt.legend(frameon=False)
plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/stimuli/cross_correlation', dpi=300)
plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/stimuli/cross_correlation.svg')


