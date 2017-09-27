#construct the covariance matrix and get eigenvectors and values
import numpy as np
cov_mat = np.cov(x_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

#plot the variance explained ratio to show how the first dimensions
#account for the variance of the matrix
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse = True)]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt
plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.show()

#proceed with feature transformation
#sort the eigenvectors based on their magnitudes
eigen_pairs = [(np.abs(eigen_vals[i]), 
                eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

#next collect the eigenvectors that correspond to the two
#largest values (60% of the variance). The number of chosen
#vectors will be a trade off between computational efficiency
#and classifier performance
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

#now transform a sample feature vector x, using the projection
#matrix onto the new feature subspace
x_train_std[0].dot(w)

#similarly transform the entire training dataset onto the new 
#subspace using the dot product
x_train_pca = x_train_std.dot(w)

#now visualize the transformation with a scatter plot
colors  = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(x_train_pca[y_train==l, 0],
                x_train_pca[y_train==l, 1],
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
"""show the plot"""