import numpy as np

class NeuralNet(object):
    def __init__(self, hidden_size, eta, alpha):
        self.hidden_size = hidden_size
        self.eta         = eta
        self.alpha       = alpha
        self.initialize_weights()

    def initialize_weights(self):
        self.w_in_hidden  = np.random.uniform(-.05, .05, (784, self.hidden_size))
        self.w_hidden_out = np.random.uniform(-.05, .05, (hidden_size, 10))
        self.b_in_hidden  = np.random.uniform(-.05, .05)
        self.b_hidden_out = np.random.uniform(-.05, .05)
      
    def sigmoid(self, xi):
        return 1.0 / (1.0 + np.exp(-zv))
                                   
    def learn(self, X_train, y_train, X_test = None, y_test = None):
        
        for xi, tar in zip(X_train, y_train):
            # forward propagate
            # input to hidden
            z2 = xi.dot(self.w_in_hidden) + self.b_in_hidden          
            a2 = sigmoid(z2)

            #hidden to output
            z3 = a2.dot(self.w_hidden_out) + self.b_hidden_out
            a3 = sigmoid(z3)

