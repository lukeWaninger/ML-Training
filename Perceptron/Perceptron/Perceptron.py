import numpy as np
from numpy.random import seed
import os

class Perceptron(object):   
    def __init__(self, 
                 y_train, 
                 train_size = 10000, 
                 test_size  = 100, 
                 iw_bounds  = [-1,1]):
       self.iw_bounds    = iw_bounds
       self.train_size   = train_size
       self.test_size    = test_size

    def pre(self, xi):
        """ Uses the current set of weights to predict the class label
        of the provided sample

        Parameters
        -----------
        xi : ndarray, float : a 1X785 array to represent the sample
        to be predicted

        Returns
        -----------
        int : the predicted class label
        """
        return np.argmax(self.w.T.dot(xi))

    def acc(self, X, t):
        """ Determines the prediction accuracy of the model
        with the current set of weights

        Parameters
        ----------
        X : ndarray : sample mX785 matrix where m is the number
        samples to predict
        t : ndarray : mX1 matrix where m is the number of samples.
        This array contains the target values for each sample
        """
        correct = 0
        for xi, ti in zip(X, t):
            if ti[0] == self.pre(xi): 
                correct += 1
        return (correct/len(X))

    def learn(self, X_train, y_train, eta, des_conv_pt, X_test = None, y_test = None):  
        """ Fits the perceptron's weights to the training data supplied

        Parameters
        ----------
        X_train : ndarray, float : a Mx784 matrix of training samples. M is the number of samples
        y_train : ndarray, float : a Mx1 matrix of target values per training sample
        eta : float : the learning rate
        des_conv_pt : float : a difference of loss at which to stop the training cycles
        X_test : ndarray, float : a Mx784 matrix of test samples. M is the number of samples
        y_test : ndarray, float : a Mx1 matrix of target values per training sample

        Returns
        ---------
        train_acc, test_acc, loss : list : each element corresponds to the accuracy/loss
        predicted over that epoch
        """
        train_acc, test_acc, loss = [], [], [0, 1]
        
        # initialize the weights as 785xN matrix to random weights between
        # o and 1 with uniform distribution where N is the number of class labels
        num_class_labels = len(np.unique(y_train))
        self.w = np.random.uniform(self.iw_bounds[0], 
                              self.iw_bounds[1], 
                              (X_train.shape[1] + 1, num_class_labels))
        
        # insert 1 into the head of each sample vector to account
        # for x_0 and the added w_0 bias
        X_train = np.insert(X_train, 0, 1, axis = 1)        
        if X_test is not None:
            X_test = np.insert(X_test, 0, 1, axis = 1)

        epoch = 1
        stop_cost = des_conv_pt + 1
        # continue training the model until the SSE has converged
        while stop_cost > des_conv_pt:
            if epoch > 70: break # just stop if the model hasn't learned by now
            error = []


            # calculate tk, yk, and dw. Then update the weights if the 
            # prediction is wrong.
            for tar, xi in zip(y_train, X_train):
                zv = xi.dot(self.w)
                tk, yk = 0, 0
                i = np.argmax(zv) # the predicted class
                tar = tar[0]      # extract from ndarray

                # yk
                if zv[i] > 0:
                    yk = 1
                # tk
                if  i == tar:
                    tk = 1
                else:
                     dw = eta * (tk - yk) * xi # perceptron learning rule
                     self.w.T[i]   += dw       # only update the weight vector that incorrectly predicted
                     self.w.T[tar] -= dw       # and the weight vector that should have predicted

                error.append(tk - yk) # for calculating the average loss for the epoch

            # append the SSE for this epoch
            loss.append((1/(2*len(X_train)))*np.sum([err**2 for err in error]))

            # find the average cost over the past three epochs
            stop_cost = (np.abs(loss[-1] - loss[-2]) + np.abs(loss[-2] - loss[-3]))/2

            # calculate prediction accuracy for training and test sets
            if X_test is not None and y_test is not None:
                train_acc.append(self.acc(X_train, y_train))
                test_acc.append(self.acc(X_test, y_test))

            # misc stuff for research            
            os.system('cls')
            print("train_size %d; test_size %d" % (self.train_size, self.test_size))
            print("initial_weight_bounds [%.2f, %.2f]\neta %f\n" % (self.iw_bounds[0], self.iw_bounds[1], eta))
            print("epoch %d\ntrain_acc %.2f; test_acc %.2f" % (epoch, train_acc[-1], test_acc[-1]))
            print("loss %.3f\tdif_last %f" % (loss[-1], stop_cost))
            epoch += 1

        return train_acc, test_acc, loss[2:] 