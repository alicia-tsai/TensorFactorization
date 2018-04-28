import numpy as np
import pandas as pd

def prepare_data_main(filename, test_indexes, first_occur_indexes):

    # ========== Helper function for preprocessing ========== #
    # read rawdata
    def read_rawdata(filename):
        df = pd.read_csv(filename)
        df.reset_index(inplace=True, drop=True)

        return df

    # prepare data dictionary
    def get_data_dict(df, first_occur_indexes):
        data = df.copy()
        d_index = data.columns.tolist().index('d_index')
        p_index = data.columns.tolist().index('p_index')
        t_index = data.columns.tolist().index('t_index')
        predict = data.columns.tolist().index('sales_amount')
        price = data.columns.tolist().index('price')

        # create dictionary
        data_dict = {}
        for row in data.values:
            key = tuple([row[d_index], row[p_index], int(row[t_index])])
            # insert 0/1 to indicate first occur
            if key in first_occur_indexes:
                data_dict[key] = [row[predict], row[price], 1]
            else:
                data_dict[key] = [row[predict], row[price], 0]

        return data_dict

    # get testing data dictionary based on indexes
    def get_test_data_dict(data_dict, test_indexes):
        test_data_dict = dict()
        for index in test_indexes:
            test_data_dict[tuple(index)] = data_dict[tuple(index)]

        return test_data_dict

    # remove testing data and data from future time from training data
    def remove_test_from_train(train_data, test_data):
        test_tuple_list = []
        for key in test_data.keys():
            test_tuple = (key[0], key[1],)
            test_tuple_list.append(test_tuple)

        month = test_data.keys()[0][2]
        for key in train_data.keys():
            train_tuple = (key[0],key[1],)
            if train_tuple in test_tuple_list or key[2] >= month:
                del train_data[key]

        return train_data

    # normalization
    def normalize(train_data, test_data, std=False):
        train_mean = np.mean(np.array(train_data.values())[:, 0])
        train_std = np.std(np.array(train_data.values())[:, 0])

        for key in train_data.keys():
            if std:
                train_data[key][0] = (train_data[key][0] - train_mean) / train_std
            else:
                train_data[key][0] = (train_data[key][0] - train_mean)

        for key in test_data.keys():
            if std:
                test_data[key][0] = (test_data[key][0] - train_mean) / train_std
            else:
                test_data[key][0] = (test_data[key][0] - train_mean)

        print('mean of training set: %.3f' %train_mean)
        print('std of training set: %.3f' %train_std)

        return train_data, test_data
    # =========================================================== #

    # ========== process data ========== #
    rawdata = read_rawdata(filename)

    train_data = get_data_dict(rawdata, first_occur_indexes)
    test_data = get_test_data_dict(train_data, test_indexes)   # get test data

    train_data = remove_test_from_train(train_data, test_data) # remove all (d,p,t) indexes in test from train data
    #train_data, test_data = normalize(train_data, test_data)   # normalize by deducting mean

    return train_data, test_data


def get_data(filename, first_occur_indexes, testing_time=False, test_indexes=False):
    if testing_time:
        test_indexes = []
        for i in first_occur_indexes:
            if i[2] == testing_time:
                test_indexes.append(list(i))
    else:
        testing_time = test_indexes[0][2]

    train_data, test_data = prepare_data_main(filename, test_indexes, first_occur_indexes)

    # make sure testing month is not in training data
    for key in train_data.keys():
        if key[2] in [m for m in range(testing_time, 103)]:
            print('found testing data in training data', key)

    return train_data, test_data
