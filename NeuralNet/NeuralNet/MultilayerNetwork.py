# Author: Daniel Eynis
# ML HW2: Multilayer Neural Network

import pandas
import numpy as np


class MultilayerNetwork:

    def __init__(self, hidden_nodes, learn_rate, momentum):
        self.hidden_nodes = hidden_nodes
        self.hidden_output_w = np.random.uniform(-0.5, 0.5, size=(self.hidden_nodes + 1, 10))
        self.input_hidden_w = np.random.uniform(-0.5, 0.5, size=(785, self.hidden_nodes))
        self.learn_rate = learn_rate
        self.momentum = momentum

    def load_data(self, train_data_path, test_data_path):
        # read in csv and convert to numpy matrix
        self.test_data = pandas.read_csv(test_data_path, header=None).as_matrix()
        self.train_data = pandas.read_csv(train_data_path, header=None).as_matrix()

        # extract target data for test and train telling which number it is
        self.test_target = self.test_data[:, 0]
        self.test_target = self.test_target.reshape(1, len(self.test_target))
        self.train_target = self.train_data[:, 0]
        self.train_target = self.train_target.reshape(1, len(self.train_target))

        # extract pixel data excluding classification data which is first column
        self.test_data = self.test_data[:, 1:]
        self.train_data = self.train_data[:, 1:]

        # scale each input value to be [0, 1]
        self.test_data = self.test_data / 255
        self.train_data = self.train_data / 255

        # insert the bias column of 1s to beginning of input array
        self.test_data = np.insert(self.test_data, 0, 1, axis=1)
        self.train_data = np.insert(self.train_data, 0, 1, axis=1)

    def learn(self):
        prev_delta_ho_w = np.zeros((10, self.hidden_nodes + 1))
        prev_delta_ih_w = np.zeros((self.hidden_nodes + 1, 785))
        for k in range(0, 50):
            for m in range(0, self.test_target.shape[1]):
                data = self.train_data[m]
                data = data.reshape(1, len(data))
                target = self.train_target[0][m]
                
                hidden_activation = 1.0 / (1.0 + np.exp(-data.dot(self.input_hidden_w)))
                hidden_activation_wbias = np.insert(hidden_activation, 0, 1, axis=1)
                output_activation = 1.0 / (1.0 + np.exp(-hidden_activation_wbias.dot(self.hidden_output_w)))
                
                target_array = np.full((1, 10), 0.1)
                target_array[0][target] = 0.9

                error_output = np.multiply((np.multiply(output_activation, (1-output_activation))), (target_array-output_activation))
                error_hidden = np.multiply((np.multiply(hidden_activation, (1-hidden_activation))), (error_output.dot(self.hidden_output_w.T[:, 1:])))
                
                for i in range(0, error_output.shape[1]):
                    delta_ho_w = (self.learn_rate*error_output[0][i]*hidden_activation_wbias[0]) + (self.momentum*prev_delta_ho_w[i])
                    prev_delta_ho_w[i] = delta_ho_w
                    self.hidden_output_w.T[i] += prev_delta_ho_w[i]
                
                for j in range(0, error_hidden.shape[1]):
                    delta_ih_w = (self.learn_rate*error_hidden[0][j]*data[0]) + (self.momentum*prev_delta_ih_w[j])
                    prev_delta_ih_w[j] = delta_ih_w
                    self.input_hidden_w.T[j] += delta_ih_w
            print('Epoch ', k, ': ', self.test_accuracy())

    def test_accuracy(self):
        num_correct = 0
        for i in range(0, self.test_target.shape[1]):
            data = self.test_data[i]
            data = data.reshape(1, len(data))
            hidden_activation = 1.0 / (1.0 + np.exp(-data.dot(self.input_hidden_w)))
            hidden_activation_wbias = np.insert(hidden_activation, 0, 1, axis=1)
            output_activation = 1.0 / (1.0 + np.exp(-hidden_activation_wbias.dot(self.hidden_output_w)))
            if np.argmax(output_activation) == self.test_target[0][i]:
                num_correct += 1
        return num_correct/self.test_target.shape[1]
