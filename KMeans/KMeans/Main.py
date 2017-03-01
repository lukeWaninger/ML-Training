import pandas as pd
import numpy  as np
import KMeans
import sys

def main():
    # read in the data
    X_train = pd.read_csv("optdigits.train").values
    X_test  = pd.read_csv("optdigits.test").values

    # experiment one
    clfs    = [KMeans.KMeans(X_train, 10) for i in range(5)]
    for clf in clfs: clf.iterate()
    best_km = clfs[np.argmin([c.mss()] for c in clfs)] 

    print()
     
if __name__ == "__main__":
    sys.exit(int(main() or 0))