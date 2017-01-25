import numpy as np

class NeuralNet(object):
    def __init__(self, sizes, iw_bounds = [-.05, .05]):
        """ 
        Parameters
        -----------
        sizes : list : length must be >= 2. 
                       Element 0: size of input
                       Elements 1 - (L - 1): sizes of hidden layers,
                         L - 2 is number of hidden layers,
                       Element -1: size of output layer
        iw_bounds : 2-tuple : represents initial weight bounds for initialization
        """
        self.sizes     = sizes
        self.L         = len(sizes)
        self.iw_bounds = iw_bounds
        self.w = []
        self._initialize_weights()

    def _initialize_weights(self):
        a, b = self.iw_bounds[0], self.iw_bounds[1]
        for s, l in zip(self.sizes[:-1], range(self.L)):
            self.w.append(np.random.uniform(a, b, (s + 1, self.sizes[l+1])))

    def _update_weights(self, a, a2, s2, s3):
        return 0

    def _sigmoid(self, z):
        """ Sigmoid activation function """
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_prime(self, z):
        """ the derivative of the sigmoid function """
        s = self._sigmoid(z)
        return s * (1 - s)

    def _forward_propagate(self, xi):
        """ moves data through layers of the network """
        z, a = [0], [xi] # started z with 0 to line up the indices with a[]
        for w in self.w:
            z = a[-1].dot(w)
            z.append(z)
            a.append(self._sigmoid(z))

        return z, a

    def _output_error(self, a3, z3, t):
        return np.multiply((t - a3), _sigmoid_prime(z3))

    def _back_propagate(self, h, o, t):
        """ compute the derivative of the cost functions to determine
        which way our weights should change for descent """
        # first compute the partial derivatives for the second weight 
        # matrix (hidden_layer_size X 1)... by chain rule
        # shows back propagation error that each weight contributed from
        # the hidden to output layers
        s = [ok * (1 - ok) * (tk - ok)]
        for j in range(self.L - 1, 1, 1):
            sj = h[j] * (1 - h[j])
            #TODO RIGHT HERE
        return s

    def _shuffle(self, X, y):
        """ shuffle the training data
        use in order to retrieve a random sample for each
        calculation. will prevent the same sample point from
        coming up and cycling the descent at that single point.
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def learn(self, X, y, eta, alpha):
        self.loss_, train_acc, test_acc = [], [], []

        for i in range(self.epochs_):
            e_loss = []
            if self.shuffle_:
                X, y = self._shuffle(X, y)

            for xi, t in zip(X, y):
                z, a = self._forward_propagate(xi)

                # find the target and adjust tk and ok vector
                # according to the homework assignment
                o = np.argmax(a[-1])
                t = np.zeros_like((10,1))
                t = 0.1
                if o == target:
                    t[o] = .9
                
                # calculate SSE across output layer
                e_loss.append(0.5 * np.sum([(tk - ok)**2 for tk, ok in zip(t, a[-1])]))

                # back-propagate
                dw = self._back_propagate(a, o, t)
                self._update_weights(dw)

        return self

    def predict(self, xi):
        a = self._forward_propagate(xi)
        return np.argmax(a, axis = 0)
        