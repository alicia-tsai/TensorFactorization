import tensorflow as tf
import numpy as np

# Define TensorFactorization model
class TensorFact:

    def __init__(self, **kwargs):
        """ Get model parameters and set tensorflow variables."""

        # get model parameters or set default value
        self.d_num = kwargs['d_num']
        self.p_num = kwargs['p_num']
        self.t_num = kwargs['t_num']

        self.set_dp_vectors = kwargs.get('set_pd_vectors', 0)
        self.set_dt_vectors = kwargs.get('set_dt_vectors', 0)
        self.set_pt_vectors = kwargs.get('set_pt_vectors', 0)

        self.set_bias_device = kwargs.get('set_bias_device', 0)
        self.set_bias_product = kwargs.get('set_bias_product', 0)
        self.set_bias_time = kwargs.get('set_bias_time', 0)
        self.set_bias = kwargs.get('set_bias', 0)
        self.set_price = kwargs.get('set_price', 0)
        self.set_first = kwargs.get('set_first', 0)

        self.lambda_cost = kwargs.get('lambda_cost', 1)
        self.lambda_d_vector = kwargs.get('lambda_d_vector', 1)
        self.lambda_p_vector = kwargs.get('lambda_p_vector', 1)
        self.lambda_t_vector = kwargs.get('lambda_t_vector', 1)
        self.lambda_temporal = kwargs.get('lambda_temporal', 10)
        self.lambda_temporal_bias = kwargs.get('lambda_temporal_bias', 0)
        self.lambda_non_negative = kwargs.get('lambda_non_negative', 0)
        self.lambda_W_price = kwargs.get('lambda_W_price', 0)
        self.lambda_W_first = kwargs.get('lambda_W_first', 0)

        self.initial_mean = kwargs.get('initial_mean', 0)
        self.initial_std = kwargs.get('initial_std', 1)
        self.laten_factor = kwargs.get('laten_factor', 16)
        self.seed = kwargs.get('seed', None)
        self.outputfile = kwargs.get('outputfile', None)

        # set tensorflow variables
        self.set_variables(self.initial_mean, self.initial_std, self.laten_factor, self.seed)

        # variables to record minimum error and final predictions
        self.min_MAE = 10000
        self.min_MSE = 10000
        self.min_epoch = 0
        self.final_predicts = None

        # create list to record error
        self.total_loss = []        # a list to store training loss for each epoch
        self.total_error_test = []   # a list to store testing error for each epoch
        self.total_error_train = []  # a list to store training error for each epoch

    def set_variables(self, initial_mean, initial_std, latent_factor, seed):
        """ Define tensorflow variables according to model parameters"""
        with open(self.outputfile, "a") as text_file: text_file.write("\nStart setting tf variables...")
        print('Start setting tf variables...')

        # set reandom seed
        if seed: tf.set_random_seed(seed)

        # create input placeholders
        self.D_indexes = tf.placeholder(tf.int32, [None])
        self.P_indexes = tf.placeholder(tf.int32, [None])
        self.T_indexes = tf.placeholder(tf.int32, [None])
        self.price = tf.placeholder(tf.float32, [None])
        self.first = tf.placeholder(tf.float32, [None])

        # create tensor weights variables
        self.D = tf.Variable(tf.random_normal([self.d_num, latent_factor], mean=initial_mean, stddev=initial_std))
        self.P = tf.Variable(tf.random_normal([self.p_num, latent_factor], mean=initial_mean, stddev=initial_std))
        self.T = tf.Variable(tf.random_normal([self.t_num, latent_factor], mean=initial_mean, stddev=initial_std))
        self.W_price = tf.Variable(tf.random_normal([1], stddev=initial_std))
        self.W_first = tf.Variable(tf.random_normal([1], stddev=initial_std))

        # create biases variables
        self.D_biases = tf.Variable(tf.random_normal([self.d_num], stddev=initial_std))
        self.P_biases = tf.Variable(tf.random_normal([self.p_num], stddev=initial_std))
        self.T_biases = tf.Variable(tf.random_normal([self.t_num], stddev=initial_std))
        self.bias = tf.Variable(tf.random_normal([1], stddev=initial_std))

        # gather tensor vectors for training
        self.d_vectors = tf.gather(self.D, self.D_indexes)
        self.p_vectors = tf.gather(self.P, self.P_indexes)
        self.t_vectors = tf.gather(self.T, self.T_indexes)
        self.t1_vectors = tf.gather(self.T, (self.T_indexes + 1)%12)

        # gather bias for training
        self.bias_device = tf.gather(self.D_biases, self.D_indexes)
        self.bias_product = tf.gather(self.P_biases, self.P_indexes)
        self.bias_time = tf.gather(self.T_biases, self.T_indexes)
        self.bias_time1 = tf.gather(self.T_biases, (self.T_indexes + 1)%12)

        # create target answers placeholders
        self.targets = tf.placeholder(tf.float32, [None])

        # define prediction function
        self.tensor_predict = self.prediction_function()

        # define cost function
        self.cost = self.cost_function()


    # ===== Helper functions for variables setting ===== #
    def prediction_function(self):
        """ Define prediction function according to model parameters setting."""
        tensor_predict = (tf.reduce_sum(self.d_vectors * self.p_vectors * self.t_vectors, axis=1)
                          + self.set_dp_vectors * tf.reduce_sum(self.d_vectors * self.p_vectors, axis=1)
                          + self.set_dt_vectors * tf.reduce_sum(self.d_vectors * self.t_vectors, axis=1)
                          + self.set_pt_vectors * tf.reduce_sum(self.p_vectors * self.t_vectors, axis=1)
                          + self.set_bias_device * self.bias_device
                          + self.set_bias_product * self.bias_product
                          + self.set_bias_time * self.bias_time
                          + self.set_bias * self.bias
                          + self.set_price * self.W_price * self.price
                          + self.set_first * self.W_first * self.first)

        return tensor_predict


    def cost_function(self):
        """ Define cost function according to model parameters setting."""
        #cost = (tf.reduce_sum(self.lambda_cost * tf.square(self.targets - self.tensor_predict)
        #                     + self.lambda_d_vector * tf.reduce_sum(tf.square(self.d_vectors), axis=1)
        #                     + self.lambda_p_vector * tf.reduce_sum(tf.square(self.p_vectors), axis=1)
        #                     + self.lambda_t_vector * tf.reduce_sum(tf.square(self.t_vectors), axis=1)
        #                     + self.lambda_temporal * tf.reduce_sum(tf.square(self.t_vectors - self.t1_vectors), axis=1)
        #                     + self.lambda_temporal_bias * tf.square(self.bias_time - self.bias_time1)
        #                     + self.lambda_non_negative * (tf.abs(self.tensor_predict) - (self.tensor_predict))
        #                     + self.lambda_W_price * tf.abs(self.W_price)
        #                     + self.lambda_W_first * tf.abs(self.W_first)))

        cost = (tf.reduce_sum(tf.reduce_sum(self.lambda_cost * tf.square(self.targets - self.tensor_predict))
                     + self.lambda_d_vector * tf.reduce_sum(tf.square(self.d_vectors))
                     + self.lambda_p_vector * tf.reduce_sum(tf.square(self.p_vectors))
                     + self.lambda_t_vector * tf.reduce_sum(tf.square(self.t_vectors))
                     + self.lambda_temporal * tf.reduce_sum(tf.square(self.t_vectors - self.t1_vectors))
                     + self.lambda_temporal_bias * tf.square(self.bias_time - self.bias_time1)
                     + self.lambda_non_negative * (tf.abs(self.tensor_predict) - (self.tensor_predict))
                     + self.lambda_W_price * tf.square(self.W_price)
                     + self.lambda_W_first * tf.square(self.W_first)))

        return cost
    # ================================================== #

    def compute_test_loss(self, test_data):
        """ Compute testing loss."""

        # get testing indexes
        indexes = np.array(test_data.keys())
        batch_index = indexes[:]
        batch_d_index = batch_index[:, 0]
        batch_p_index = batch_index[:, 1]
        batch_t_index = batch_index[:, 2]

        target_values = []       # ground truth answer
        price_values = []
        first_occur_values = []
        for key in tuple(map(tuple,batch_index)):
            target_values.append(test_data[key][0])
            price_values.append(test_data[key][1])
            first_occur_values.append(test_data[key][2])


        feed_dict = {
            self.D_indexes: batch_d_index,
            self.P_indexes: batch_p_index,
            self.T_indexes: batch_t_index,
            self.price: price_values,
            self.first: first_occur_values
        }

        # get predictions
        predicts = self.sess.run(self.tensor_predict, feed_dict=feed_dict)

        # compute error
        error_SE = (target_values - predicts)**2
        error_AE = np.absolute(target_values - predicts)

        return np.mean(error_SE), np.mean(error_AE)


    def get_final_prediction(self, test_data):
        """ Get final testing prediction values."""

        # get testing indexes
        indexes = np.array(test_data.keys())
        batch_index = indexes[:]
        batch_d_index = batch_index[:, 0]
        batch_p_index = batch_index[:, 1]
        batch_t_index = batch_index[:, 2]

        price_values = []
        first_occur_values = []
        for key in tuple(map(tuple,batch_index)):
            price_values.append(test_data[key][1])
            first_occur_values.append(test_data[key][2])

        feed_dict = {
            self.D_indexes: batch_d_index,
            self.P_indexes: batch_p_index,
            self.T_indexes: batch_t_index,
            self.price: price_values,
            self.first: first_occur_values
        }

        # get predictions
        final_predicts = self.sess.run(self.tensor_predict, feed_dict=feed_dict)

        return final_predicts


    def set_model_parameters(self, gup_memory_fraction=0.1):
        """ Initialize tensorflow variables and model parameters."""
        with open(self.outputfile, "a") as text_file: text_file.write("\nStart initializing parameters...")
        print('Start initializing parameters...')

        # define Optimizer
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(self.cost)

        # initialize all variables
        self.init = tf.global_variables_initializer()

        # GPU version
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gup_memory_fraction)
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = tf.Session()

        self.sess.run(self.init)


    def train_model(self, train_data, test_data, learning_rate=1, batch_size=1000, n_epochs=500,
                    learning_rate_decay=False, print_per_epoch=10, initial=True):
        """ Main training procedure. Create a tensor factorizatin model."""

        # init parameters
        if initial: self.set_model_parameters()

        # start training
        with open(self.outputfile, "a") as text_file: text_file.write("\nStart training model...")
        print("Start training model...")

        for epoch in range(n_epochs):
            indexes = np.array(train_data.keys())
            np.random.shuffle(indexes)
            if learning_rate_decay:
                self.learning_rate = learning_rate if epoch < (n_epochs//2) else learning_rate*0.01
            else:
                self.learning_rate = learning_rate

            batch_index = indexes[:]
            batch_d_index = batch_index[:, 0]
            batch_p_index = batch_index[:, 1]
            batch_t_index = batch_index[:, 2]

            target_values = []
            price_values = []
            first_occur_values = []
            for key in tuple(map(tuple,batch_index)):
                target_values.append(train_data[key][0])
                price_values.append(train_data[key][1])
                first_occur_values.append(train_data[key][2])

            feed_dict = {
                self.D_indexes: batch_d_index,
                self.P_indexes: batch_p_index,
                self.T_indexes: batch_t_index,
                self.price: price_values,
                self.first: first_occur_values,
                self.targets: target_values,
                self.learning_rate_placeholder: self.learning_rate
            }

            loss, predicts, _ = self.sess.run([self.cost, self.tensor_predict, self.optimizer], feed_dict=feed_dict)

            # record training and testing error
            test_error_MSE, test_error_MAE = self.compute_test_loss(test_data)
            train_error_MSE = np.mean((target_values - predicts)**2)
            train_error_MAE = np.mean(np.absolute(target_values - predicts))

            self.total_error_test.append([test_error_MSE, test_error_MAE])
            self.total_error_train.append([train_error_MSE, train_error_MAE])
            self.total_loss.append(loss)

            if epoch % print_per_epoch == 0:
                with open(self.outputfile, "a") as text_file:
                        text_file.write("\n\tepoch %d; train MSE: %f; train MAE: %f; test MSE: %f; test MAE: %f" \
                                        %(epoch+1, train_error_MSE, train_error_MAE, test_error_MSE, test_error_MAE))
                print("\tepoch %d; train MSE: %f; train MAE: %f; test MSE: %f; test MAE: %f" \
                            % (epoch+1, train_error_MSE, train_error_MAE, test_error_MSE, test_error_MAE))

            # record minimum error and final predictions
            if test_error_MAE < self.min_MAE:
                self.min_MAE = test_error_MAE
                self.min_MSE = test_error_MSE
                self.min_epoch = epoch
                self.final_predicts = self.get_final_prediction(test_data)


        # get final error
        with open(self.outputfile, "a") as text_file:
                text_file.write("\n\tepoch %d; train MSE: %f; train MAE: %f; test MSE: %f; test MAE: %f" \
                                %(n_epochs, self.total_error_train[-1][0], self.total_error_train[-1][1],\
                                  self.total_error_test[-1][0], self.total_error_test[-1][1]))
        print("\tepoch %d; train MSE: %f; train MAE: %f; test MSE: %f; test MAE: %f" \
                  %(n_epochs, self.total_error_train[-1][0], self.total_error_train[-1][1], \
                    self.total_error_test[-1][0], self.total_error_test[-1][1]))

        # get minimum error
        with open(self.outputfile, "a") as text_file:
                text_file.write("\nMinimum epoch %d; test MSE %f; test MAE %f" %(self.min_epoch, self.min_MSE, self.min_MAE))
                text_file.write("\nTraining Finished!")
        print("Minimum epoch %d; test MSE %f; test MAE %f" %(self.min_epoch, self.min_MSE, self.min_MAE))
        print('Training Finished!')


    def close_session(self):
        self.sess.close()
