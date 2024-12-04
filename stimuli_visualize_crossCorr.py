import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# select directory to save stimuli
root            = '/prednet_Brands2024_git/'
data_save       = '/prednet_Brands2024_git/data/stimuli/img_statistics/'

# datasets
datasets            = ['KITTI', 'WT_AMS', 'WT_VEN', 'WT_WL']
datasets_lbls       = ['KITTI', 'Amsterdam', 'Venice', 'Wild life']
color               = ['#F6C141', '#4EB265', '#5289C7', '#DC050C']

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
        data_current = np.load(data_save + 'crossCorr_' + dataset + '_sample.npy')

        print(data_current)

        # compute spread
        data_mean   = np.mean(data_current)
        data_sem    = np.std(data_current)

        # visualize
        axs.plot([offset_iD[iD], offset_iD[iD]], [data_mean - data_sem, data_mean + data_sem], color=color[iD], zorder=-1)
        axs.scatter(offset_iD[iD], data_mean, color=color[iD], edgecolor='white', s=160)
        # axs.scatter(np.ones(len(data_current))*offset_iD[iD], data_current, color=color[iD], s=1, alpha=0.25)

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


