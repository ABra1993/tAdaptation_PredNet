import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import yaml
from sklearn.utils import resample
from lgnpy.lgnpy.CEandSC.lgn_statistics import lgn_statistics
import scipy

# input shape for PredNet
input_shape = [128, 160, 3]
if input_shape[0] == 120:
    data_save = '/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/'
elif input_shape[0] == 128:
    data_save = '/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/'

# select directory to save stimuli
root            = '/home/amber/OneDrive/code/prednet_Brands2024/'
config_path     = '/home/amber/OneDrive/code/prednet_Brands2024/lgnpy/lgnpy/CEandSC/default_config.yml'

# categories
cats            = ['bodies', 'buildings', 'faces', 'objects', 'scenes', 'scrambled']
idx_scenes      = 4

# img info
dataset         = 'set2'
img_n           = 24
cat_idx         = 4 # SCENES INDEX

# initiate dataframe to store CE and SC values
CEandSC_values = np.zeros((img_n, 2))

# configure
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)
print(config)

# se threshold
threshold_lgn = scipy.io.loadmat('/home/amber/OneDrive/code/prednet_Brands2024/lgnpy/ThresholdLGN.mat')['ThresholdLGN']

# compute values
for iImg in range(img_n):
# for iImg in range(10, 11):

    # print progress
    print('Processing img ', iImg+1, '/', img_n)

    # import from set1
    img = Image.open(root + 'stimuli/' + dataset + '/img' + str(cat_idx*img_n+iImg+1) + '.jpg')
    img = np.array(img)
    print(img.shape)

    # plot image
    fig = plt.figure()
    plt.imshow(img)
    plt.savefig(root + 'visualization/stimuli/img_statistics/' + dataset + '/' + dataset + '_img' + str(iImg+1))
    plt.axis('off')
    plt.close()

    # compute CE and SC
    ce, sc, _, _ = lgn_statistics(im=np.array(img), file_name=str(iImg+1), config=config, force_recompute=True, cache=False, home_path=config_path, threshold_lgn=threshold_lgn)
    
    # save values
    CEandSC_values[iImg, 0] = ce[:, 0, 0].mean()
    CEandSC_values[iImg, 1] = sc[:, 0, 0].mean()

    print(CEandSC_values[iImg, 0])
    print(CEandSC_values[iImg, 1])

# save array
print(CEandSC_values)
# np.save(data_save + dataset, CEandSC_values)


# CEandSC_values = np.load('/home/amber/Documents/prednet_Brands2024/data/stimuli/img_statistics/' + dataset_stim + '.npy')

# print(CEandSC_values)

# idx_min = np.argmin(CEandSC_values[:, 0])
# print('Min (CE):', int(idx_min+1+96))

# idx_max = np.argmax(CEandSC_values[:, 0])
# print('Max (CE): ', int(idx_max+1+96))

# idx_min = np.argmin(np.delete(CEandSC_values[:, 1], 10))
# print('Min (SC):', int(idx_min+1+96))

# idx_max = np.argmax(np.delete(CEandSC_values[:, 1], 10))
# print('Max (SC): ', int(idx_max+1+96))