# generate the data
import numpy as np
from sklearn.datasets import make_circles
X, y = make_circles(n_samples    = 1000, 
                    random_state = 123, 
                    noise        = 0.1,
                    factor       = 0.2)

import matplotlib.pyplot as plt
plt.clf()
plt.scatter(X[y == 0, 0], 
            X[y == 0, 1],
            color  = 'red',
            marker = '^',
            alpha  = 0.5)
plt.scatter(X[y == 1, 0], 
            X[y == 1, 1],
            color  = 'blue',
            marker = 'o',
            alpha  = 0.5)
""" show the plot"""

# start with standard PCA approach to compare with RBF
from sklearn.decomposition import PCA
plt.clf()
sc_pca  = PCA(n_components = 2)
X_spca  = sc_pca.fit_transform(X)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
ax[0].scatter(X_spca[y == 0, 0], 
              X_spca[y == 0, 1],
              color  = 'red',
              marker = '^',
              alpha  = 0.5)
ax[0].scatter(X_spca[y == 1, 0], 
              X_spca[y == 1, 1],
              color  = 'blue',
              marker = 'o',
              alpha  = 0.5)
ax[1].scatter(X_spca[y == 0, 0], 
              np.zeros((500,1)) + 0.02, #this vertical shift is to show the class overlap
              color  = 'red',
              marker = '^',
              alpha  = 0.5)
ax[1].scatter(X_spca[y == 1, 0], 
              np.zeros((500,1)) - 0.02, #this vertical shift is to show the class overlap
              color  = 'blue',
              marker = 'o',
              alpha  = 0.5)
""" show the plot """

# now for comparison separate via RBF kernel
import rbf as rbf_
X_kpca = rbf_.rbf_kernel_pca(X, gamma = 15, n_components = 2)
plt.clf()
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
ax[0].scatter(X_kpca[y == 0, 0], 
              X_spca[y == 0, 1],
              color  = 'red',
              marker = '^',
              alpha  = 0.5)
ax[0].scatter(X_kpca[y == 1, 0], 
              X_spca[y == 1, 1],
              color  = 'blue',
              marker = 'o',
              alpha  = 0.5)
ax[1].scatter(X_kpca[y == 0, 0], 
              np.zeros((500,1)) + 0.02, #this vertical shift is to show the class overlap
              color  = 'red',
              marker = '^',
              alpha  = 0.5)
ax[1].scatter(X_kpca[y == 1, 0], 
              np.zeros((500,1)) - 0.02, #this vertical shift is to show the class overlap
              color  = 'blue',
              marker = 'o',
              alpha  = 0.5)
""" show the plot """