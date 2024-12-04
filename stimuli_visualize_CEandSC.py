import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from neural_data_visualize_utils import *
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def getImage(path):
    return OffsetImage(plt.imread(path), zoom=.05, alpha = 1)

# set root
root        = '/home/amber/OneDrive/code/prednet_Brands/' 

# stimulus dataset
dataset_stim = ['set1', 'set2']

# select model
model = 'Lotter2017'
# model = 'Kirubeswaran2023'

# # training data
# dataset = 'FPSI'
# dataset = 'single_seq'

# number of images
n_img = 48

# set directory's
if model == 'Lotter2017':
    root_data       = '/home/amber/Documents/prednet_Brands2024/data/model/Lotter2017/'
    root_vis        = '/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Lotter2017/twopulse/'
elif model == 'Kirubeswaran2023':
    root_data       = '/home/amber/Documents/prednet_Brands2024/data/model/Kirubeswaran2023/avgs/'
    root_vis        = '/home/amber/OneDrive/code/prednet_Brands2024/visualization/model/Kirubeswaran2023/'

root_data_save  = '/home/amber/Documents/prednet_Brands2024/data/metrics/onepulse/'

# import CE and SC values
CEandSC_values = np.zeros((n_img, 2))
for i, datasetstim in enumerate(dataset_stim):
    if i == 0:
        CEandSC_values[:int(n_img/2)] = np.load('/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/' + datasetstim + '.npy')
    else:
        CEandSC_values[int(n_img/2):] = np.load('/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/' + datasetstim + '.npy')

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
        file_path = '/home/amber/OneDrive/code/prednet_Brands2024/stimuli/set1/img' + str(96 + int(index % (n_img/2) + 1)) + '.jpg'
    else:
        file_path = '/home/amber/OneDrive/code/prednet_Brands2024/stimuli/set2/img' + str(96 + int(index % (n_img/2) + 1)) + '.jpg'
    
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
plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/stimuli/img_statistics/CEandSCvalues', dpi=300)
plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/stimuli/img_statistics/CEandSCvalues.svg')

