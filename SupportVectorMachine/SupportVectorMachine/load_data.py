from sklearn.cross_validation   import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.preprocessing      import MinMaxScaler
import pandas                   as pd
import os, sys, collections

def main():
    # read in the data
    data = pd.read_csv("spambase.data", header = None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

    # standardize everything
    stdsc   = StandardScaler()
    X_train = stdsc.fit_transform(X_train)
    X_test  = stdsc.transform(X_test)

    pd.to_hdf([X_train, X_test, y_train, y_test], "spambase.hdf")

if __name__ == "__main__":
    sys.exit(int(main() or 0))