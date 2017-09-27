# load the dataset
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

# identify class labels
from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y) # the array of class labels
le.transform(['M', 'B'])

# split into test/training sets 20/80
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size = 0.20, random_state = 1)