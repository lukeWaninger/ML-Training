import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)

#preprocessing
#separate into 70/30 training/test splits and standardize to unit variance
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0:].values
x_train, x_test, y_train, y_test = \
    train_test_split(X, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std  = sc.transform(x_test)