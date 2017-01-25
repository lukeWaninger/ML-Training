import numpy as np

class NeuralNetwork(object):
    def __init__(self, input, output = 10,
                 hidden = 10, eta = 1e-3, des_conv_pt = 1e-4,
                 iw_bounds = [-.05, .05], epochs = 1, shuffle = True):
        self.input_layer_size_  = input
        self.hidden_layer_size_ = hidden
        self.output_layer_size_ = output
        self.eta_               = eta
        self.des_acc_           = des_conv_pt
        self.epochs_            = epochs
        self.shuffle_           = True
        self.iw_bounds          = iw_bounds
        self.w = []
        self._initialize_weights()
        self.b = np.zeros_like((1,2))

    def _initialize_weights(self):
        # w2 generates an hidden_layer_size X input_layer_size matrix
        # where each weights is random and uniformly distributed between
        # the input bounds
        w2 = np.random.uniform(self.iw_bounds[0], self.iw_bounds[1], 
                              size = self.hidden_layer_size_*(self.input_layer_size_))
        w2 = w2.reshape(self.hidden_layer_size_, self.input_layer_size_)
        # w3 generates an output_layer_size X hidde_layer size matrix
        # matrix of weights with the same pattern
        w3 = np.random.uniform(self.iw_bounds[0], self.iw_bounds[1], 
                              size = self.output_layer_size_ * (self.hidden_layer_size_))
        w3 = w3.reshape(self.output_layer_size_, self.hidden_layer_size_)
        self.w.append(w2)
        self.w.append(w3)
        self.w_initialized = True

    def _update_weights(self, a, a2, s2, s3):
        c = self.eta_/len(a)
        self.w[0] -= c*np.sum(s2.dot(a.T))
        self.w[1] -= c*np.sum(s3.dot(a2))
        self.b[0] -= c*np.sum(s2)
        self.b[1] -= c*np.sum(s3)

    def _sigmoid(self, z):
        """ Sigmoid activation function """
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_prime(self, z):
        """ the derivative of the sigmoid function """
        s = self._sigmoid(z)
        return s * (1 - s)

    def _forward_propagate(self, X):
        """ moves data through layers of the network """
        a1 = X
        z2 = self.w[0].dot(a1.T) + self.b[0]
        a2 = self._sigmoid(z2) 
        z3 = self.w[1].dot(a2) + self.b[1]
        a3 = self._sigmoid(z3)

        return a1, z2, a2, z3, a3

    def _output_error(self, a3, z3, t):
        return np.multiply((t - a3), _sigmoid_prime(z3))

    def _quadratic_cost(self, output, target):
        """ function returns the cost using l2 methods
        the sum of the squared difference between real and actual"""
        return (1/(2*len(output)))*np.sum((target - np.argmax(output))**2)

    def _back_propagate(self, a1, t, z2, a2, z3, a3):
        """ compute the derivative of the cost functions to determine
        which way our weights should change for descent """
        # first compute the partial derivatives for the second weight 
        # matrix (hidden_layer_size X 1)... by chain rule
        # shows back propagation error that each weight contributed from
        # the hidden to output layers
        s3 = (a3.T - t)
        z2 -= self.b[1]
        s2 = np.multiply(self.w[1].T.dot(s3.T), self._sigmoid_prime(z2))
        dw1 = s2.dot(a1)
        dw2 = s3.T.dot(a2.T)

        return dw1, dw2

    def _shuffle(self, X, y):
        """ shuffle the training data
        use in order to retrieve a random sample for each
        calculation. will prevent the same sample point from
        coming up and cycling the descent at that single point.
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def learn(self, X, y):
        self.cost_ = []

        for i in range(self.epochs_):
            if self.shuffle_:
                X, y = self._shuffle(X, y)

            a1, z2, a2, z3, a3 = self._forward_propagate(X)
            self.cost_.append(self._quadratic_cost(a3, y))
            s2, s3 = self._back_propagate(X, y, z2, a2, z3, a3)
            self._update_weights(a1, a2, s2, s3)

        return self

    def predict(self, X):
        prop = self._forward_propagate(X)
        return prop[-1][np.argmax(prop[-2], axis = 0)]
        