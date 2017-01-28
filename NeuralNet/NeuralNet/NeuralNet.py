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
        self.w = [(0)]
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

        # for each hidden layer, apply the weights (with bias as w[0] and activate
        for wl, l in zip(self.w[1:], range(self.L)):
            a[l] = self.reshape(np.insert(a[l], 0, 1))  # insert a one for the bias
            zv = a[l].T.dot(wl)                 # returns a (hidden-layer-size + 1,) vector
            z.append(zv)                      

            hl = self.sigmoid(zv)              # apply the activation function
            a.append(hl)
        return z, a

    def back_propagate(self, a, t, z):
        """ compute the derivative of the cost functions to determine gradient """
        # 1. calculate the output error
        s     = [0 for i in range(self.L)]
        s[-1] = ((a[-1] - t.T) * self.sigmoid_prime(z[-1])).T

        # reshape each z vector to account for the bias
        #for l in range(1, self.L, 1):
        #    z[l] = self.reshape(np.insert(z[l], 0, 1))

        # 2. calculate the error for each hidden layer
        for l in range(self.L - 2, -1, -1):
            # TODO: VERIFY MATRIX SIZES ARE CORRECT AS THIS FUNCTION PASSES BACKWARDS
            s[l] = np.multiply((self.w[l + 1][1:].dot(s[l + 1])), self.sigmoid_prime(z[l]))
        return s

    def shuffle(self, X, y):
        """ shuffle the training data
        use in order to retrieve a random sample for each
        calculation. will prevent the same sample point from
        coming up and cycling the descent at that single point.
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def reshape(self, m):
        return m.reshape([m.shape[0], 1])

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
                t = self.reshape(np.array([.1 for i in range(self.sizes[-1])]))
                if o == tar[0]:
                    t[o] = .9
                
                # calculate SSE across output layer
                e_loss.append(0.5 * np.sum([(tk - ok)**2 for tk, ok in zip(t, a[-1])]))

                # back-propagate
                dw = self.back_propagate(a, t, z)
                self.update_weights(dw)

        return self

    def predict(self, xi):
        a = self.forward_propagate(xi)
        return np.argmax(a, axis = 0)
        