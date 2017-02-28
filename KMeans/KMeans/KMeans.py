import numpy as np

class KMeans(object):
    def __init__(self):
            self.K = 10

    def mse(self, C):
        return np.average([d(x, centeroid(C)) for x in C])

    def avg_mse(self, C):
        return np.sum([mse(c) for c in C])/len(C)

    def mss(self):
        return 0

    def entropy(self):
        return 0

    def mean_entropy(self):
        return 0

    def d(self, x, y):
        return np.sqrt(np.sum(x-y)**2)

    def centeroid(self, C):
        return 0