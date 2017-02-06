import numpy  as np
import pandas as pd
from sklearn.cross_validation   import train_test_split
from sklearn.preprocessing      import StandardScaler
import sys, collections

def main():
    # read in the data
    data = pd.read_csv("spambase.data", header = None)
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 123)

    # standardize everything
    stdsc   = StandardScaler()
    X_train = stdsc.fit_transform(X_train)
    X_test  = stdsc.fit_transform(X_test)

    # show class distribution
    print(collections.Counter(X_train))
    print(collections.Counter(X_trest))



if __name__ == "__main__":
    sys.exit(int(main() or 0))