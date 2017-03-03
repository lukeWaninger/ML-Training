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

    def fit(self):
        self.clusters_designated = False
        prev = []
        while True:  
            # reset sample containment
            for c in self.C:
                c[1] = []
        
            # add each sample to its closest centeroid     
            for xi in self.X_train:
                #TODO: BREAK TIES AT RANDOM
                self.C[np.argmin([self.d(xi[:-1], c[0]) for c in self.C])][1].insert(0, xi)
        
            # move the centeroid
            prev = [c[0] for c in self.C]
            for c in self.C:
                if len(c[1]) == 0: continue
                c[0] = np.array([np.mean(fi) for fi in (np.array(c[1])[:,:-1]).T])
            distance = np.abs(np.sum([self.d(np.array(p), np.array(c[0])) for p, c in zip(prev, self.C)]))
            
            # break the loop if the centeroids stop moving
            if distance < .001: break
            os.system('cls') 
            print(distance)
            
    def mse(self, C):
        if len(C[1]) == 0: return None
        return np.mean([self.d(x[:-1], C[0]) for x in C[1]])

    def avg_mse(self):
        sum, count = 0, 1
        for c in self.C:
            if len(c[1]) == 0: continue
            sum += self.mse(c)
            count += 1
        return sum/count # TODO: change this to only divide by the nonzero Ks

    def mss(self):     
        sum = 0
        for i in range(len(self.C)):
            for j in range(i + 1, len(self.C)):
                sum += self.d(self.C[i][0], self.C[j][0])
        return sum/((self.K*(self.K-1))/2)

    def d(self, x, y):
        return np.sqrt(np.sum((x-y)**2))

    def pred(self, xi):
        # assign classes to centers based on mode
        if not self.clusters_designated:
            for c in self.C:
                if len(c[1]) == 0: continue
                mode = stats.mode(np.array(c[1])[:,-1])
                c[2] = mode[0][np.random.randint(0, len(mode[1]))]
            self.clusters_designated = True   

        distances = [self.d(xi[:-1], c[0]) for c in self.C]        
        return self.C[np.argmin(distances)][2]