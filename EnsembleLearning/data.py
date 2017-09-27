from sklearn                    import datasets
from sklearn.cross_validation   import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.preprocessing      import LabelEncoder

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size = 0.5, random_state = 1)