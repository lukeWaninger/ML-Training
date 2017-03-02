import numpy as np
import scipy.stats as stats
import os

class KMeans(object):
    def __init__(self, X_train, K = 2):
       self.K = K
       # self.C is a list of 3 tuples in the form of a list because why in the world
       # did the designers of this language make a tuple immutable where each tuple 
       # represents a cluster with ([centeroid], [samples_i,...,samples_m], assigned class)
       self.X_train = X_train
       self.C = [[np.array([np.random.randint(0, 16) for j in range(X_train.shape[1] - 1)]), np.array([]), -np.inf] for i in range(K)]
       self.clusters_designated = False

    """
    first pick a K, the best K would be a K = to the number of classes.
    find the euclidean distances between points and centeroids
    assign points to centers
    move centers
    repeat until centeroids stop moving
    """
    def fit(self):     
        prev = []
        while True:  
            # reset sample containment
            for c in self.C:
                c[1] = []
        
            # add each sample to its closest centeroid     
            for xi in self.X_train:
                self.C[np.argmin([self.d(xi[:-1], c[0]) for c in self.C])][1].insert(0, xi[:-1])
        
            # move the centeroid
            prev = [c[0] for c in self.C]
            for c in self.C:
                if len(c[1]) == 0: continue
                c[0] = np.array([np.mean(fi) for fi in np.array(c[1]).T])
            distance = np.abs(np.sum([self.d(np.array(p), np.array(c[0])) for p, c in zip(prev, self.C)]))
            if distance < 1: break
            os.system('cls')
            print(distance)
            
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
        if not self.clusters_designated:
            for c in self.C:
                mode = stats.mode(np.array(c[1])[:,-1])[0][0]
            self.clusters_designated = True   

        # find distances and return the set of predictions
        y_pred = []
        for xi in X_test:
            y_pred.append(self.C[np.argmin([self.d(xi[:-1], c[0]) for c in self.C])][2])
        return y_pred