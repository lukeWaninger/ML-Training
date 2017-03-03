import numpy as np
import scipy.stats as stats
import os

class KMeans(object):
    def __init__(self, X_train, K = 2, conv_pt = .001):
       """
       KMeans
       Parameters
       -----------
       X_train : array-like : training data set such that the last element in each row is the target class
       K : int : number of clusters
       conv_pt : flot : iteration ends if sum of distance moved for all centeroids from one iteration to the next is less
       """       
       # self.C is a list of 3 tuples in the form of a list because why in the world
       # did the designers of this language make a tuple immutable? Each tuple 
       # represents a cluster with ([centeroid], [samples_i,...,samples_m], assigned class)
       self.K       = K
       self.X_train = X_train
       self.C       = [[np.array([np.random.randint(0, 16) for j in range(X_train.shape[1] - 1)]), np.array([]), -np.inf] for i in range(K)]
       self.conv_pt = conv_pt
       self.clusters_designated = False

    def fit(self):
        self.clusters_designated = False
        prev = []
        while True:  
            # reset sample containment
            for c in self.C:
                c[1] = []
        
            # add each sample to its closest centeroid, breaking ties at random  
            for xi in self.X_train:
                distances = np.array([self.d(xi[:-1], c[0]) for c in self.C])
                indices   = np.argwhere(distances == distances.min())
                selection = indices[np.random.randint(0, len(indices))][0]
                self.C[selection][1].insert(0, xi)
        
            # move the centeroid            
            for c in self.C:
                if len(c[1]) == 0: continue
                c[0] = np.array([np.mean(fi) for fi in (np.array(c[1])[:,:-1]).T])
            distance_last = np.abs(np.sum([self.d(np.array(p), np.array(c[0])) for p, c in zip(prev, self.C)]))
            prev.append(distance_last)

            # break the loop if the centeroids stop moving
            if distance_last < self.conv_pt : break
            # determine oscillation
            if len(distances) > 8:
                mean = np.mean(prev[-8:])
                sum  = np.sum(prev[-8:] - mean)
                if sum < conv_pt: break

            os.system('cls') 
            print(distance_last)
            
    def mse(self, C):
        if len(C[1]) == 0: return None
        return np.mean([self.d(x[:-1], C[0]) for x in C[1]])

    def avg_mse(self):
        sum, count = 0, 1
        for c in self.C:
            if len(c[1]) == 0: continue
            sum += self.mse(c)
            count += 1
        return sum/count

    def mss(self):     
        sum = 0
        for i in range(len(self.C)):
            for j in range(i + 1, len(self.C)):
                sum += self.d(self.C[i][0], self.C[j][0])
        return sum/((self.K*(self.K-1))/2)

    def d(self, x, y):
        return np.sqrt(np.sum((x-y)**2))

    def pred(self, xi):
        # assign classes to centers based on mode, break ties at random
        if not self.clusters_designated:
            for c in self.C:
                if len(c[1]) == 0: continue
                mode = stats.mode(np.array(c[1])[:,-1])
                c[2] = mode[0][np.random.randint(0, len(mode[1]))]
            self.clusters_designated = True   

        distances = [self.d(xi[:-1], c[0]) for c in self.C]        
        return self.C[np.argmin(distances)][2]