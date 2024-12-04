import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from neural_data_visualize_utils import *
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def getImage(path):
    return OffsetImage(plt.imread(path), zoom=.01, alpha = 1)

# set root
root        = '/prednet_Brands2024_git/' 

# number of images
n_img = 1600

# set directory's
root            = 'prednet_Brands2024_git'
root_data       = 'datasets/Groen2013/'

# import CE and SC values
CE = np.loadtxt('datasets/Groen2013/model_CE.txt', delimiter=',')
SC = np.loadtxt('datasets/Groen2013/model_SC.txt', delimiter=',')

# create pandas dataframe
df = pd.DataFrame(columns=['img', 'CE', 'SC'])
df['img'] = np.arange(n_img)+1
df['CE'] = CE
df['SC'] = SC
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

    # import image
    if index < 9:
        file_path = root_data + 'im_000' + str(index+1) + '.jpg'
    elif index < 99:
        file_path = root_data + 'im_00' + str(index+1) + '.jpg'
    elif index < 999:
        file_path = root_data + 'im_0' + str(index+1) + '.jpg'
    else:
        file_path = root_data + 'im_' + str(index+1) + '.jpg'
    
    # add to plot
    ab = AnnotationBbox(getImage(file_path), (row['SC'], row['CE']), frameon=False)
    ax.add_artist(ab)

# adjust axes
ax.tick_params(axis='both', labelsize=fontsize_tick)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('CE', fontsize=fontsize_label)
ax.set_xlabel('SC', fontsize=fontsize_label)
ax.set_xlim(0.2, 1.9)
ax.set_ylim(-0.1, 1.6)

# save figure
plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/stimuli/img_statistics/CEandSCvalues_Groen2013', dpi=300)
plt.savefig('/home/amber/OneDrive/code/prednet_Brands2024/visualization/stimuli/img_statistics/CEandSCvalues_Groen2013.svg')
