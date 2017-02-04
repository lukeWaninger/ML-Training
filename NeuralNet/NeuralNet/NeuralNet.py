import numpy as np
import os

class NeuralNet(object):
    def __init__(self, input_size, hidden_size, eta, alpha):
        self.eta    = eta
        self.alpha  = alpha
        self.initialize_weights(input_size, hidden_size)

    def initialize_weights(self, input_size, hidden_size):
        bound = [-0.05, 0.05]
        self.w_in_hidden  = np.random.uniform(bound[0], bound[1], (input_size + 1, hidden_size))
        self.w_hidden_out = np.random.uniform(bound[0], bound[1], (hidden_size + 1, 10))
   
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
                                   
    def learn(self, X_train, y_train, X_test = None, y_test = None):
        X = np.insert(X_train, 0, 1, axis = 1)
        test_acc, train_acc = [], []

        # train for the range of epochs
        for e in range(10):
            # arrays to store the previous weight change (for momentum)
            p_dwk = np.zeros_like(self.w_hidden_out)
            p_dwj = np.zeros_like(self.w_in_hidden)  

            # loop through every training example
            for xi, tar in zip(X, y_train):
                # forward propagate
                # -----------------
                # input to hidden
                z2 = xi.dot(self.w_in_hidden)
                a2 = self.sigmoid(z2)
                a2_wbias = np.insert(a2, 0, 1, axis = 0)

                # hidden to output
                z3 = a2_wbias.dot(self.w_hidden_out)
                a3 = self.sigmoid(z3)

                # back-propagation
                # ----------------------
                # generate target vector
                t = np.full((1, 10), 0.1)
                t[0][tar[0]] = .9

                # calculate error at output and hidden layers
                err_out = a3 * (1 - a3) * (t - a3)
                err_hid = a2 * (1 - a2) * err_out.dot(self.w_hidden_out.T[:,1:])

                # update weights
                # ---------------------------------------
                # update the hidden->output weight matrix
                for wkj, sk, p_dwkj, i \
                in zip(self.w_hidden_out.T, err_out.T, p_dwk.T, range(0, p_dwk.shape[1] - 1, 1)):
                    dwkj = self.eta * sk * a2_wbias + self.alpha * p_dwkj
                    p_dwk.T[i] = dwkj
                self.w_hidden_out += p_dwk

                # update the input->hidden weight matrix
                for wji, sj, p_dwji, i \
                in zip(self.w_in_hidden.T, err_hid.T, p_dwj.T, range(0, p_dwj.shape[1] - 1, 1)):
                    dwji = self.eta * sj * xi + self.alpha * p_dwji
                    p_dwj.T[i] = dwji
                self.w_in_hidden += p_dwj

            # calculate accuracy
            if X_test is not None and y_test is not None:
                train_acc.append(self.predict(X_train, y_train))
                test_acc.append(self.predict(X_test, y_test))

            # print some info for each epoch
            os.system('cls')
            print("Epoch: %d" % e)
            print("Training accuracy: %.2f" % train_acc[-1])
            print("Testing accuracy:  %.2f" % test_acc[-1]) 

        return train_acc, test_acc

    def predict(self, X, y = None):
        if y is not None:
            X = np.insert(X, 0, 1, axis = 1)
            pos = 0
            for xi, tar in zip(X, y):
                output_vector = np.insert(self.sigmoid(xi.dot(self.w_in_hidden)), 0, 1, axis = 0)
                output_vector = self.sigmoid(output_vector.dot(self.w_hidden_out))
            
                if np.argmax(output_vector) == tar[0]:
                    pos += 1
            return (pos/X.shape[0]) * 100
        else:
            X = np.insert(X, 0, 1, axis = 0)
            output_vector = np.insert(self.sigmoid(X.dot(self.w_in_hidden)), 0, 1, axis = 0)
            output_vector = self.sigmoid(output_vector.dot(self.w_hidden_out))
            return np.argmax(output_vector)