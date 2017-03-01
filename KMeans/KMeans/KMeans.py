import numpy as np
import scipy.stats as stats

class KMeans(object):
    def __init__(self, X_train, K = 2):
       self.K = K
       # self.C is a list of 3 tuples where each tuple represents a 
       # cluster with ([centeroid], [samples_i,...,samples_m], assigned class)
       self.C = [([np.random.randint(0, 16) for j in range(X_train.shape[1] - 1)], [], -np.inf) for i in range(K)]

    """
    first pick a K, the best K would be a K = to the number of classes.
    find the euclidean distances between points and centeroids
    assign points to centers
    move centers
    repeat until centeroids stop moving
    """
    def iterate(self):     
        while True:  
            # reset sample containment
            for c in self.C:
                c[2] = []
        
            # add each sample to its closest centeroid     
            for xi in self.X_train:
                self.C[np.argmin([d(xi, c) for c in self.C])][2].append(xi)
        
            # move the centeroid storing the previous center to
            # determine whether or not to break the loop
            previous_c = [c[0] for c in self.C]
            for c in self.C:
                c[0] = [np.mean(fi) for fi in c[2]]
            if np.sum([d(p, c[0]) for p, c in zip(previous_c, self.C)]) < 1: break
            
    def mse(self, C):
        return np.average([d(x, centeroid(self.C)) for x in self.C])

    def avg_mse(self, C):
        return np.sum([mse(c) for c in C])/len(C)

    def mss(self):
        sum1 = np.sum([d(self.C[i][0], self.C[j][0]) for i, j in zip(range(len(self.C)), range (i + 1, len(self.C)))])
        sum = 0
        for i in range(len(self.C)):
            for j in range(i + 1, len(self.C)):
                sum += d(self.C[i][0], self.C[j][0])
        return sum/((self.K*(K-1))/2)

    def d(self, x, y):
        return np.sqrt(np.sum((x-y)**2))

    def pred(self, X_test):
        # assign classes to centers based on mode
        for c in self.C:
            mode = stats.mode(c[2][:,-1])
            # TODO: BREAK THIS TIE AT RANDOM