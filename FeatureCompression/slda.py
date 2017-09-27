import matplotlib.pyplot as plt
plt.clf()

from sklearn.lda import LDA
lda = LDA(n_components = 2)
x_train_lda = lda.fit_transform(x_train_std, y_train)

# use logistic regression with the scikit's linear-discriminant-analysis
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(x_train_lda, y_train)

plot_decision_regions(x_train_lda, y_train, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower left')
""" show the plot """

# perform the same with the test set
x_test_lda = lda.transform(x_test_std)
plot_decision_regions(x_test_lda, y_test, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower left')

# which correctly classifies are samples using only the
# the new 2-dimensional subspace generated via LDA.