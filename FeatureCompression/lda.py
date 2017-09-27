import numpy as np
np.set_printoptions(precision=4)

# first compute the mean vectors from the original feature space
mean_vecs = []
for label in range (1,4):
    mean_vecs.append(np.mean(x_train_std[y_train == label], axis = 0))
    print('MV %s: %s\n' %(label, mean_vecs[label - 1]))

# next compute the within-class scatter matrix
d = 13 # number of features
s_w = np.zeros((d, d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.zeros((d, d))
    
    for row in x_train[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)

    s_w += class_scatter

print('Within-class scatter matrix: %sx%s' % (s_w.shape[0], s_w.shape[1]))

# we are making the assumption that the data is normally distributed
# but show the distribution via
print('class label distribution: %s' % np.bincount(y_train)[1:])

# which shows the class labels are not normally distributed so we 
# can get a normalized scatter matrix by computing the covariance matrix
# which just so happens to be the same as a normalized scatter matrix
d = 13 # number of features
s_w = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(x_train_std[y_train == label].T)
    s_w += class_scatter

print('scaled within-class scatter matrix: %sx%s' % (s_w.shape[0], s_w.shape[1]))

# now compute the between-class scatter matrix
mean_overall = np.mean(x_train_std, axis = 0)
d = 13 # number of features
s_b = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):
    n = x_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    
s_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print('between-class scatter matrix: %sx%s' % (s_b.shape[0], s_b.shape[1]))

# compute the dot product of ((s_w)^-1) and s_b and find its eigenvectors and eigenvalues
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))

# sort the eigenvalues in decreasing order so we can find the
# features that have the highest discriminatory capabilities
eigen_pairs = [(np.abs(eigen_vals[i]), 
                eigen_vecs[:,i]) for i in range(len(eigen_vals))]

print('eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

# now measure how much class discrimination can be captured by the vectors
import matplotlib.pyplot as plt

tot = sum(eigen_vals.reals)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse = True)]
cum_disc = np.cumsum(discr)
plt.bar(range(1, 14), 
        discr,
        alpha = 0.5, 
        align = 'center',
        label = 'individual "discriminability"')
plt.step(range(1, 14),
         cum_disc,
         where = 'mid',
         label = 'cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('linear discriminants')
ply.ylim([-0.1, 1.1])
plt.legend(loc='best')
""" show the plot """

# now stack the most discriminative eigenvector columns to create W
w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real, eigen_pairs[1][1][:,np.newaxis].real))
print('matrix W:\n', w)

# now project onto the new feature space
x_train_lda = x_train_std.dot(w)
colors  = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(x_train_lda[y_train == 1, 0]*(-1),
                x_train_lda[y_train == 1, 1]*(-1),
                c = c, label = l, marker = m)

    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc = 'lower right')
""" show the plot """