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

    def update_weights(self, error, a, eta, alpha):
        # update weights that feed the output layer
        for i in range(error[2].shape[0]):
            dw = eta * error[2] * a[1][i]
            c = self.reshape(self.w[2][i]) + dw
            c = c.reshape([c.shape[0],])
            self.w[2][i] = c

        # update weights that feed the hidden layer
        for i in range(error[1][1:].shape[0]):
            dw = eta * error[1][1:] * a[0][i]
            c = self.reshape(self.w[1][i]) + dw
            c = c.reshape([c.shape[0],])
            self.w[1][i] = c
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
        s     = []
        s.append(np.multiply((np.multiply(a[-1], (1 - a[-1]))), (t - a[-1])))

        # 2. calculate the error for the hidden layer
        dw = self.sigmoid_prime(a[1])
        s[0] = error_hidden = np.multiply((np.multiply(a[1], (1-a[1]))), (s[1].dot(a[1].T[:, 1:])))
        
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

    def learn(self, X, y, eta, alpha, X_test = None, y_test = None):
        self.loss, train_acc, test_acc = [1], [], []

        epoch = 1
        while epoch < 10 or self.loss[-1] > 1e-2:
            e_loss = []
            if self.shuffle:
                X, y = self.shuffle(X, y)

            for xi, tar in zip(X, y):
                z, a = self.forward_propagate(xi)

                # find the target and adjust tk and ok vector
                # according to the homework assignment
                o = np.argmax(a[-1])
                t = np.full((1, 10), 0.1)
                t[0][o] = .9
                
                # calculate SSE across output layer
                e_loss.append(0.5 * np.sum([(tk - ok)**2 for tk, ok in zip(t, a[-1])]))

                # back-propagate
                dw = self.back_propagate(a, t, z)
                self.update_weights(dw, a, eta, alpha)
            
            if X_test is not None and y_test is not None:
                 train_acc.append(self.accuracy(X, y))
                 test_acc.append(self.accuracy(X_test, y_test))

        return self

    def accuracy(self, X, y):
        for xi, tar in zip(X, y):
                count = 0
                p = self.predict(xi)
                if p == tar[0]:
                    count += 1
        return (count/len(X))

    def predict(self, xi):
        a = self.forward_propagate(xi)
        p = np.argmax(a[-1][-1], axis = 1)[0]
        return p      