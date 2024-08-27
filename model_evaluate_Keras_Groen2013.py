import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input
from prednet_Keras import PredNet

# set root
root            = '/home/amber/Documents/prednet_Brands2024/'
root_data       = '/home/amber/Documents/prednet_Brands2024/data/stimuli/Groen2013/'

def main():

    ########################## NETWORK SETTINGS

    
    # network settings
    batch_size          = 100
    n_img               = 1600
    n_batch             = int(n_img/batch_size)

    n_layer             = 4
    nt                  = 45

    # input shape for PredNet
    input_shape = [128, 160, 3]

    # import data
    trial = 'onepulse'
    shape               = (n_img, nt, input_shape[2], input_shape[0], input_shape[1])
    X                   = np.memmap(root_data + 'imgs_' + trial, dtype='float32', mode='r+', shape=shape)

    # output mode
    output_modes = ['E']
    for output_mode in output_modes:
        assert output_mode in ['error', 'prediction', 'E', 'R', 'A', 'Ahat']

    # trained or random initialization
    trained = True

    print('output mode: ', output_modes)
    print('trained: ', trained)
    print('trial: ', trial)

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

        for iL in range(n_layer):

            # print progress
            print('Layer: L', iL+1)

            # initiate dataframe
            metrics_avg = np.zeros((n_img, nt))
            print('Output shape: ', metrics_avg.shape)

            for i in range(n_batch):

                # print progress
                print('Batch ', i+1, '/', n_batch, '...')

                # select data
                X_test = X[i*batch_size:i*batch_size+batch_size, :, :, :, :]

                # fig, axs = plt.subplots(1, nt)
                # for t in range(nt):
                #     axs[t].imshow(np.transpose(X_test[0, t, :, :, :], (1, 2, 0)))
                #     axs[t].axis('off')
                # plt.show()

                # add to configuration
                layer_config['output_mode'] = output_mode + str(iL)

                # initiate model
                if trained:
                    test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
                else:
                    test_prednet = PredNet(**layer_config)

                # model grouping layers into an object with training/inference features
                predictions = test_prednet(inputs)
                test_model = Model(inputs=inputs, outputs=predictions)

                # predict model
                X_hat = test_model.predict(X_test, batch_size)
                metrics_avg[i*batch_size:i*batch_size+batch_size, :] = X_hat.reshape(X_hat.shape[0], X_hat.shape[1], X_hat.shape[2]*X_hat.shape[3]*X_hat.shape[4]).mean(2)

                # save
                if trained:
                    np.save('/home/amber/Documents/prednet_Brands2024/data/model/Lotter2017/Groen2013/avgs/prednet_onepulse_' + output_mode + str(iL+1) + '_actvs_trained', metrics_avg)
                else:
                    np.save('/home/amber/Documents/prednet_Brands2024/data/model/Lotter2017/Groen2013/avgs/prednet_onepulse_' + output_mode + str(iL+1) + '_actvs_random', metrics_avg)

if __name__ == '__main__':
    main()
