plot_decision_regions(X = X_combined_std, y = y_combined, classifier = ppn, test_idx = range(105,150)) 
plt.xlabel('petal length [standardized]') 
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
"""show plot"""

"""using the logistic regression model for the cost function"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]') 
plt.legend(loc ='upper left')
"""Show plot"""

"""to show conditional probabilities of the sample prediction correctness"""

lr.predict_proba(X_test_std[0,:])

"""maximum margin classification with the slack variable and C optimization"""
plt.clf()

from sklearn.svm import SVC
svm = SVC(kernel='linear',C=1.0,random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
"""Show the plot"""

"""Solving nonlinear problems using kernel SVM
genearate a random XOR dataset to show how a nonlinear
classification problem looks"""

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.clf()
plt.scatter(X_xor[ y_xor == 1, 0], X_xor[y_xor == 1, 1], c ='b', marker ='x', label ='1')
plt.scatter(X_xor[ y_xor ==-1, 0], X_xor[y_xor ==-1, 1], c ='r', marker ='s', label ='-1') 
plt.ylim(-3.0)
plt.legend()
"""Show the plot"""

"""define a Radial Bias Function kernel (RBF) on random Xor data"""
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.10,C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier = svm)
plt.legend(loc ='upper left') 
"""Show the plot"""


"""now use the RBF function with our flower shiz"""

"""assign petal length and petal width of the 150 flower
samples to the feature matrix X and corresponding class labels
to the vector y"""

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

"""split dataset into training and test sets"""

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

"""standardize the features using scikit preprocessing module"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)
y = df.iloc[0:100, 4].values 
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test)) 
svm = SVC(kernel='rbf', random_state=0, gamma=10.10,C=10.0)
svm.fit(X_train_std, y_train)
plt.clf()
plot_decision_regions(X_combined_std, y_combined, classifier = svm, test_idx = range(105,150))
plt.xlabel('petal length [standardized]') 
plt.ylabel('petal width [standardized]')
plot_decision_regions(X_combined_std, y_combined, classifier = svm)
plt.legend(loc ='upper left')
"""Show the plot"""