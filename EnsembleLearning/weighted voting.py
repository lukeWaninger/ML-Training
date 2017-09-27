import numpy as np

np.argmax(np.bincount([0, 0, 1],
                      weights = [ 0.2, 0.2, 0.6 ]))

# weighted majority vote based on class probabilities
ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]])
p = np.average(ex, axis = 0, weights = [ 0.2, 0.2, 0.6 ] )
p
np.argmax(p)