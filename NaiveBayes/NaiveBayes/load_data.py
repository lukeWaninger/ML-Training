from sklearn.cross_validation import train_test_split
from sklearn.preprocessing    import StandardScaler
import pandas                 as pd
import os, sys, collections

def main():
    # read in the data
    data = pd.read_csv("spambase.data", header = None)
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
    
    df = pd.DataFrame([X_train, y_train, X_test, y_test])
    df.to_hdf("spambase.hdf", "hw4")

if __name__ == "__main__":
    sys.exit(int(main() or 0))