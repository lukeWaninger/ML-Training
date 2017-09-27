from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# generate random data
X, y = make_moons(n_samples = 100, random_state = 123)

# initialize sklearns kernel PCA
sc_kpca = KernelPCA(n_components = 2,
                    kernel = 'rbf',
                    gamma  = 15)

# fit the data
X_skpca = sc_kpca.fit_transform(X)

# visualize the data
plt.scatter(X_skpca[y == 0, 0], 
              X_skpca[y == 0, 1],
              color  = 'red',
              marker = '^',
              alpha  = 0.5)
plt.scatter(X_skpca[y == 1, 0], 
              X_skpca[y == 1, 1],
              color  = 'blue',
              marker = 'o',
              alpha  = 0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
""" show the plot """