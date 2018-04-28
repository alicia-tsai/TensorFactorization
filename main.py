#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import collections
import pickle

from model import TensorFact

if __name__ == "__main__":

    first_occur_indexes = np.load('data_dict/first_occur.npy')
    outputfile = 'output/Output_allmonths.txt'

    # define model parameters
    model_parameters = {
        'd_num': 499,
        'p_num': 433,
        't_num': None,
        'set_dp_vectors': 0,
        'set_dt_vectors': 0,
        'set_pt_vectors': 0,
        'set_bias_device': 1,
        'set_bias_product': 1,
        'set_bias_time': 0,
        'set_bias': 1,
        'set_price': 0,
        'set_first': 0,
        'lambda_cost': 1,
        'lambda_d_vector': 0.01,
        'lambda_p_vector': 0.01,
        'lambda_t_vector': 0.01,
        'lambda_temporal': 0.003,
        'lambda_temporal_bias': 0,
        'lambda_non_negative': 1,
        'lambda_W_price': 0,
        'lambda_W_frist': 0,
        'initial_mean': 0,
        'initial_std': 1,
        'latent_factor': 16,
        'outputfile': outputfile
    }

    MSE_all = [] # store all final MSE
    MAE_all = [] # store all final MAE

    # training for month 12 to 23 (the second year)
    for month in range(12, 24, 1):
        # read data
        testing_time = month
        model_parameters['t_num'] = 12
        #train_data, test_data = get_data(filename, first_occur_indexes, testing_time=testing_time)

        # get train, test dictionary
        savefile_train = 'data_dict/train_month_' + str(testing_time) + '.txt'
        savefile_test = 'data_dict/test_month_' + str(testing_time) + '.txt'
        with open(savefile_train, "rb") as datafile:
            train_data = pickle.load(datafile)
        with open(savefile_test, "rb") as datafile:
            test_data = pickle.load(datafile)

        # change t index to only 12
        new_train_data = collections.defaultdict()
        new_test_data = collections.defaultdict()

        for key in train_data.keys():
            new_key = (int(key[0]), int(key[1]), key[2] % 12)
            new_train_data[new_key] = train_data[key]

        for key in test_data.keys():
            new_key = (int(key[0]), int(key[1]), key[2] % 12)
            new_test_data[new_key] = test_data[key]

        # run TF model
        print('\nTraining on month: ' + str(testing_time))
        TF_model = TensorFact(**model_parameters)
        TF_model.train_model(new_train_data, new_test_data, learning_rate=1e-3, n_epochs=50, \
                             print_per_epoch=50, learning_rate_decay=False)
        TF_model.close_session()

        # save testing prediction
        predictions = []
        for x, y in zip(test_data.keys(), TF_model.final_predicts):
            predictions.append(list(x))
            predictions[-1].append(y)
        savename = 'predict_month_' + str(testing_time)
        np.save('output/predict/' + savename, predictions)

        # prediction 0 for value smaller than 0
        with open(outputfile, "a") as text_file:
            final_MSE = np.mean((((TF_model.final_predicts > 0) * TF_model.final_predicts) \
                                                   - np.array(test_data.values())[:, 0])**2)
            final_MAE = np.mean(np.absolute(((TF_model.final_predicts > 0) * TF_model.final_predicts) \
                                                              - np.array(test_data.values())[:, 0]))
            MSE_all.append(final_MSE)
            MAE_all.append(final_MAE)
            text_file.write('\nprediction 0 for value smaller than 0')
            text_file.write('\nMSE: ' + str(final_MSE))
            text_file.write('\nMAE: ' + str(final_MAE))
            text_file.write('\n\n')

        # save error per epoch
        error_per_epoch = np.array([np.array(TF_model.total_error_train), np.array(TF_model.total_error_test)])
        savename = 'error_month_' + str(testing_time)
        np.save('output/error/' + savename, error_per_epoch)

    print('All training finished!')
