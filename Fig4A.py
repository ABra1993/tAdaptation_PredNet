import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def getImage(path):
    return OffsetImage(plt.imread(path), zoom=.05, alpha = 1)

# set root
root            = '/home/amber/OneDrive/code/prednet_Brands2024_git/' 

root_imgs       = '/home/amber/OneDrive/datasets/Brands2024/'
root_sumMet     = '/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/'

# stimulus dataset
dataset_stim = ['set1', 'set2']

# select model
model = 'Lotter2017'

# number of images
n_img = 48

# import CE and SC values
CEandSC_values = np.zeros((n_img, 2))
for i, datasetstim in enumerate(dataset_stim):
    if i == 0:
        CEandSC_values[:int(n_img/2)] = np.load(root_sumMet + datasetstim + '.npy')
    else:
        CEandSC_values[int(n_img/2):] = np.load(root_sumMet + datasetstim + '.npy')

# create pandas dataframe
df = pd.DataFrame(columns=['img', 'CE', 'SC'])
df['img'] = np.arange(n_img)+1
df['CE'] = CEandSC_values[:, 0]
df['SC'] = CEandSC_values[:, 1]
print(df)

# initiate figure
fig = plt.figure()
ax = plt.gca()
sns.despine(offset=10)

# fontsizes 
fontsize_title          = 20
fontsize_legend         = 12
fontsize_label          = 18
fontsize_tick           = 15

for index, row in df.iterrows():

    # get file to path
    if index < int(n_img/2):
        file_path = root_imgs + 'set1/img' + str(96 + int(index % (n_img/2) + 1)) + '.jpg'
    else:
        file_path = root_imgs + 'set2/img' + str(96 + int(index % (n_img/2) + 1)) + '.jpg'
    
    # add to plot
    ab = AnnotationBbox(getImage(file_path), (row['SC'], row['CE']), frameon=False)
    ax.add_artist(ab)

# adjust axes
ax.tick_params(axis='both', labelsize=fontsize_tick)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('CE', fontsize=fontsize_label)
ax.set_xlabel('SC', fontsize=fontsize_label)
ax.set_ylim(-0.0005, 0.008)
ax.set_xlim(0, 2.8)

# save figure
plt.tight_layout()
plt.scatter(CEandSC_values[:, 1], CEandSC_values[:, 0])
plt.savefig(root + 'visualization/Fig4A', dpi=600, bbox_inches='tight')
plt.savefig(root + 'visualization/Fig4A.svg', bbox_inches='tight')

