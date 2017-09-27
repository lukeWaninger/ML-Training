# generate random half moon dataset
import numpy as np
import rbf as rbf_
from sklearn.datasets import make_moons

X, y = make_moons(n_samples = 100, random_state = 123)
alphas, lambdas = rbf_.rbf_kernel_pca(X, gamma = 15, n_components = 1)

x_new  = X[25]       # to be used as the 'new' data point for projection
x_proj = alphas[25]  # original projection

# the function to project a new sample point
def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum(
        (x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    
    return k.dot(alphas/lambdas) 

# now project
x_reproj = project_x(x_new, 
                     X, 
                     gamma = 15, 
                     alphas = alphas, 
                     lambdas = lambdas)

# visualize the projection
import matplotlib.pyplot as plt
plt.scatter(alphas[y == 0, 0], 
              np.zeros((50)),
              color  = 'red',
              marker = '^',
              alpha  = 0.5)
plt.scatter(alphas[y == 1, 0], 
              np.zeros((50)),
              color  = 'blue',
              marker = 'o',
              alpha  = 0.5)
plt.scatter(x_proj, 
              0,
              color  = 'black',
              marker = '^',
              s      = 100,
              label  = 'original projection of point X[25]')
plt.scatter(x_reproj,
              0,
              color  = 'green',
              marker = 'x',
              s      = 500,
              label  = 'remapped point X[25]')
plt.legend(scatterpoints = 1)
""" show the plot """