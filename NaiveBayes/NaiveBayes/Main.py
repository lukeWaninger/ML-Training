from sklearn.metrics            import confusion_matrix
import matplotlib.pyplot        as plt
import sklearn.metrics          as metrics
import numpy                    as np
import pandas                   as pd
import os, sys, collections

def main():
    # read in the data
    data = pd.read_hdf("spambase.hdf", header = None)
    X_train, y_train, X_test, y_test = data.values[0][0], data.values[1][0], data.values[2][0], data.values[3][0]

    p_0 = np.bincount(y_train)[0]/y_train.shape[0]
    p_1 = 1 - p_0

    std_muw = np.array([(np.std(xi), np.mean(xi)) for xi in X_train.T])
    
   

    print(p_xi)
    
def gaussian_pdf(x,s,m):
    return (1/(np.sqrt(2*np.pi)*s)) * np.exp((-(x-m)**2/(2*s**2)))


if __name__ == "__main__":
    sys.exit(int(main() or 0))