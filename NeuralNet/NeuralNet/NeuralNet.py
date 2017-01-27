import numpy as np

class NeuralNet(object):
    def __init__(self, sizes, conv_pt = 1e2, iw_bounds = [-.05, .05]):
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
        self.initialize_weights()

    def initialize_weights(self):
        a, b = self.iw_bounds[0], self.iw_bounds[1]
        for s, l in zip(self.sizes[:-1], range(self.L)):
            wl = np.random.uniform(a, b, (s + 1, self.sizes[l+1]))
            self.w.append(wl)

    def update_weights(self, s, h, eta, alpha):
        for w, sk, hj in zip(self.w, s, h):
            # TODO
            break
        return 0

    def sigmoid(self, zv):
        """ Sigmoid activation function """
        return 1.0 / (1.0 + np.exp(-zv))

    def sigmoid_prime(self, z):
        """ the derivative of the sigmoid function """
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward_propagate(self, xi):
        """ moves data through layers of the network """
        z, a = [0], [xi] # started z with 0 to line up the indices with a[]
        for wl in self.w:
            a[-1] = np.insert(a[-1], 0, 1)
            zv = a[-1].dot(wl)  # returns a (21,)
            # should I reshape that to a (21,1) before appending it? yes
            z.append(zv)
            a.append(self.sigmoid(zv)) # do you activate the bias?

        return z, a

    def output_error(self, a3, z3, t):
        return np.multiply((t - a3), sigmoid_prime(z3))

    def back_propagate(self, h, t):
        """ compute the derivative of the cost functions to determine
        which way our weights should change for descent """
        # first compute the partial derivatives for the second weight 
        # matrix (hidden_layer_size X 1)... by chain rule
        # shows back propagation error that each weight contributed from
        # the hidden to output layers
        s = [ok * (1 - ok) * (tk - ok) for tk, ok in zip(t, h[-1])]
        for j, k in zip(range(self.L - 1, 1, 1), range(0, self.L - 1, 1)):
            sj = sigmoid_prime(h[j]) * np.sum(self.w[j+1] * s[k]) 
            s.append(sj)
        return s.sort(reverse = True)

    def shuffle(self, X, y):
        """ shuffle the training data
        use in order to retrieve a random sample for each
        calculation. will prevent the same sample point from
        coming up and cycling the descent at that single point.
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def learn(self, X, y, eta, alpha):
        self.loss, train_acc, test_acc = [1], [], []

        epoch = 1
        while epoch < 100 or self.loss[-1] > 1e-2:
            e_loss = []
            if self.shuffle:
                X, y = self.shuffle(X, y)

            for xi, tar in zip(X, y):
                z, a = self.forward_propagate(xi)

                # find the target and adjust tk and ok vector
                # according to the homework assignment
                o = np.argmax(a[-1])
                t = np.array([.1 for i in range(0, 9)])
                if o == tar[0]:
                    t[o] = .9
                
                # calculate SSE across output layer
                e_loss.append(0.5 * np.sum([(tk - ok)**2 for tk, ok in zip(t, a[-1])]))

                # back-propagate
                dw = self.back_propagate(a, t)
                self.update_weights(dw)

        return self

    def predict(self, xi):
        a = self.forward_propagate(xi)
        return np.argmax(a, axis = 0)
        