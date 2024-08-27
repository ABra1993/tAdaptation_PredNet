import os
import numpy as np
# from six.moves import cPickle
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import theano
import tensorflow as tf
import seaborn as sns
import math

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet_Lotter2017 import PredNet

dataset = 'set1'

root        = '/home/amber/Documents/prednet_Brands2024/'
root_stim   = '/home/amber/Documents/prednet_Brands2024/data/stimuli/128_160/' + dataset + '/'

def main():

    ########################## NETWORK SETTINGS
    
    # network settings
    batch_size          = 1
    n_img               = 24
    n_layer             = 4
    tempCond            = [1, 2, 4, 8, 16, 32]
    nt                  = 45

    # trial types
    # trials = np.array(['onepulse', 'twopulse_repeat', 'twopulse_nonrepeat_same', 'twopulse_nonrepeat_diff'])
    # trials = np.array(['onepulse'])
    trials = np.array(['onepulse', 'twopulse_repeat', 'twopulse_nonrepeat_same'])

    # output mode
    output_modes = ['prediction']
    for output_mode in output_modes:
        assert output_mode in ['error', 'prediction', 'E', 'R', 'A', 'Ahat']

    # trained or random initialization
    trained = True

    print('trials: ', trials)
    print('output mode: ', output_modes)
    print('trained: ', trained)

    ########################## INITIATE NETWORK

    # specify files
    weights_file    = os.path.join('/home/amber/OneDrive/code/prednet_Brands2024/weights/Lotter2017/tensorflow_weights/prednet_kitti_weights.hdf5')
    json_file       = os.path.join('/home/amber/OneDrive/code/prednet_Brands2024/weights/Lotter2017/prednet_kitti_model.json')

    # load trained model
    f = open(json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
    train_model.load_weights(weights_file)
    train_model.summary()

    # get layer configuration
    layer_config = train_model.layers[1].get_config()

    # input shape
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    print('Shape input: ', input_shape)

    ########################## INFERENCE

    for output_mode in output_modes:

        if output_mode == 'error':

            # initiate dataframe
            metrics_avg = np.zeros((len(tempCond), n_img, nt, 4)) # 4 = number of layers

            # add to configuration
            layer_config['output_mode'] = 'error'

            # initiate model
            if trained:
                test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
            else:
                test_prednet = PredNet(**layer_config)

            # model grouping layers into an object with training/inference features
            predictions = test_prednet(inputs)
            test_model = Model(inputs=inputs, outputs=predictions)

            for trial in trials:

                # print progress
                print('Trial: ', trial)
                
                # import data
                X_test = np.load(root_stim + 'stimuli_' + trial + '.npy')

                for iC in range(len(tempCond)): # iterate over temporal conditions (ISI or duration)
                # for iC in range(1):

                    # print progress
                    print('Temp cond: ', tempCond[iC])

                    # predict model
                    X_hat = test_model.predict(X_test[iC, :, :, :, :, :], batch_size)
                    metrics_avg[iC, :, :, :] = X_hat

                # save
                if trained:
                    np.save(root + 'data/model/Lotter2017/' + dataset + '/prednet_' + trial + '_' + output_mode + '_actvs_KITTI', metrics_avg)
                else:
                    np.save(root + 'data/model/Lotter2017/' + dataset + '/prednet_' + trial + '_' + output_mode + '_actvs_random', metrics_avg)

        elif output_mode == 'prediction':

            for trial in trials:

                # print progress
                print('Trial: ', trial)
                
                # import data
                X_test = np.load(root_stim + 'stimuli_' + trial + '.npy')

                # add to configuration
                layer_config['output_mode'] = output_mode

                # initiate model
                if trained:
                    test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
                else:
                    test_prednet = PredNet(**layer_config)

                # compute output shape
                output = test_prednet.compute_output_shape(X_test[0, :, :, :, :, :].shape)

                # initiate dataframe
                metrics_avg = np.zeros((len(tempCond), n_img, nt, int(output[2]), int(output[3]), int(output[4])))
                print('Output shape: ', metrics_avg.shape)

                # model grouping layers into an object with training/inference features
                predictions = test_prednet(inputs)
                test_model = Model(inputs=inputs, outputs=predictions)

                for iC in range(len(tempCond)): # iterate over temporal conditions (ISI or duration)

                    # print progress
                    print('Temp cond: ', tempCond[iC])

                    # predict model
                    X_hat = test_model.predict(X_test[iC, :, :, :, :, :], batch_size)
                    metrics_avg[iC, :, :, :, :, :] = X_hat

                # save
                if trained:
                    np.save('/home/amber/Documents/prednet_Brands2024/data/model/Lotter2017/' + dataset + '/feature_maps/prednet_' + trial + '_' + output_mode + '_actvs_KITTI', metrics_avg)
                else:
                    np.save('/home/amber/Documents/prednet_Brands2024/data/model/Lotter2017/' + dataset + '/feature_maps/prednet_' + trial + '_' + output_mode + '_actvs_random', metrics_avg)


        else:
            for trial in trials:

                # print progress
                print('Trial: ', trial)
                
                # import data
                X_test = np.load(root_stim + 'stimuli_' + trial + '.npy')

                for iL in range(n_layer):

                    # print progress
                    print('Layer: L', iL+1)

                    # add to configuration
                    layer_config['output_mode'] = output_mode + str(iL)

                    # initiate model
                    if trained:
                        test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
                    else:
                        test_prednet = PredNet(**layer_config)

                    # compute output shape
                    output = test_prednet.compute_output_shape(X_test[0, :, :, :, :, :].shape)

                    # initiate dataframe
                    metrics_avg = np.zeros((len(tempCond), n_img, nt, int(output[2]), int(output[3]), int(output[4])))
                    print('Output shape: ', metrics_avg.shape)

                    # model grouping layers into an object with training/inference features
                    predictions = test_prednet(inputs)
                    test_model = Model(inputs=inputs, outputs=predictions)

                    for iC in range(len(tempCond)): # iterate over temporal conditions (ISI or duration)

                        # print progress
                        print('Temp cond: ', tempCond[iC])

                        # predict model
                        X_hat = test_model.predict(X_test[iC, :, :, :, :, :], batch_size)
                        metrics_avg[iC, :, :, :, :, :] = X_hat

                    # save
                    if trained:
                        np.save('/home/amber/Documents/prednet_Brands2024/data/model/Lotter2017/' + dataset + '/feature_maps/prednet_' + trial + '_' + output_mode + str(iL+1) + '_actvs_KITTI', metrics_avg)
                    else:
                        np.save('/home/amber/Documents/prednet_Brands2024/data/model/Lotter2017/' + dataset + '/feature_maps/prednet_' + trial + '_' + output_mode + str(iL+1) + '_actvs_random', metrics_avg)


if __name__ == '__main__':
    main()
