"""Show the difference between costs as epochs increase with the learning
rate equal to .01 and .0001"""

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize =(8, 4))
ada1 = AdalineGD(n_iter = 10, eta = 0.01).fit(X, y)
ax[0].plot( range( 1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker ='o') 
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)') 
ax[0].set_title('Adaline - Learning rate 0.01') 
ada2 = AdalineGD(n_iter = 10, eta = 0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2. cost_) + 1),
ada2.cost_, marker ='o') 
ax[1].set_xlabel('Epochs') 
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

"""As shown, it would take an excessive number of epochs for the training data
to learn and converge on J_min with the learning rate equal to .0001 but .01
overshoots the cost min and begins to bound back up the curve. But we can 
use the gradient descent scaling properties to optimize the learning rate"""

"""This gradient descent benefits from 'scaling'. Here we can standardize
this scaling to give the data a standard normal distribution such that the
mean is 0 and standard deviation is 1. Standardizing the jth feature can be
achieved by (X_j - mean_j)/sigma_j We can do this using NumPy """

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,0] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

"""Now that we have standardized the feature vector X, a learning rate of
.01 will converge with 15 iterations"""

ada = AdalineGD(n_iter=15, eta=0.1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier = ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]') 
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left') 
"""Show the modified plot"""

plt.plot(range( 1, len( ada.cost_) + 1), ada.cost_, marker ='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
"""Show the plot"""

"""Run with the stochastic gradiant descent (SGD) to show how much
faster the cost converges to minimum"""

ada = AdalineSGD(n_iter = 15, eta = 0.01, random_state = 1)
ada.fit(X_std, y) 
plot_decision_regions(X_std, y, classifier = ada) 
plt.title('Adaline - Stochastic Gradient Descent') 
plt.xlabel('sepal length [standardized]') 
plt.ylabel('petal length [standardized]') 
plt.legend(loc ='upper left') 
"""Show the plot"""

plt.plot(range(1, len( ada.cost_) + 1), ada.cost_, marker ='o')
plt.xlabel('Epochs') 
plt.ylabel('Average Cost')
"""Show the plot"""