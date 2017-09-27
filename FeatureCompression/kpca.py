# KPCA USAGE
# generate a nonlinearly separable dataset
from sklearn.datasets import make_moons
X, y = make_moons(n_samples = 100, random_state = 123)

import matplotlib.pyplot as plt
plt.scatter(X[y == 0, 0], X[y == 0, 1], 
            color  = 'red', 
            marker = '^',
            alpha  = 0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], 
            color  = 'blue',
            marker = 'o',
            alpha  = 0.5)
""" show the plot """

# perform some PCA (unsupervised) to show separability via sklearn
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
              np.zeros((50,1)) + 0.02, #this vertical shift is to show the class overlap
              color  = 'red',
              marker = '^',
              alpha  = 0.5)
ax[1].scatter(X_spca[y == 1, 0], 
              np.zeros((50,1)) - 0.02, #this vertical shift is to show the class overlap
              color  = 'blue',
              marker = 'o',
              alpha  = 0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
""" show the plot """
# which shows the classes are now linearly separable

# project separability via PCA after using the RBF kernel
plt.clf()
from matplotlib.ticker import FormatStrFormatter
X_kpca = rbf_kernel_pca(X, gamma = 15, n_components = 2)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7,3))
ax[0].scatter(X_kpca[y == 0, 0], 
              X_kpca[y == 0, 1],
              color  = 'red',
              marker = '^',
              alpha  = 0.5)
ax[0].scatter(X_kpca[y == 1, 0], 
              X_kpca[y == 1, 1],
              color  = 'blue',
              marker = 'o',
              alpha  = 0.5)
ax[1].scatter(X_kpca[y == 0, 0], 
              np.zeros((50,1)) + 0.02, #this vertical shift is to show the class overlap
              color  = 'red',
              marker = '^',
              alpha  = 0.5)
ax[1].scatter(X_kpca[y == 1, 0], 
              np.zeros((50,1)) - 0.02, #this vertical shift is to show the class overlap
              color  = 'blue',
              marker = 'o',
              alpha  = 0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].set_major_formatter(FormatStrFormatter('%0.1f'))
""" show the plot """