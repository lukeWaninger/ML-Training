import numpy    as np
import pandas   as pd
import matplotlib.pyplot as plt
import sklearn.metrics   as metrics
from sklearn.cross_validation   import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.svm                import LinearSVC
import os, sys, collections

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
    #print(collections.Counter(X_train))
    #print(collections.Counter(X_test))
    svm = exp_one(X_train, y_train, X_test, y_test)
    exp_two(X_train, y_train, X_test, y_test, svm)

def exp_one(X_train, y_train, X_test, y_test):
    # initialize and fit the SVM
    svc = LinearSVC(random_state = 1)
    svc.fit_transform(X_train, y_train)

    # predict with the test set
    y_pred = svc.decision_function(X_test)
    print('Accuracy Score: %.3f; Precision: %.3f; Recall %.3f' %
            (metrics.accuracy_score(y_test, y_pred),
            metrics.precision_score(y_test, y_pred),
            metrics.recall_score(y_test, y_pred)))

    fpr, tpr, threshholds = metrics.roc_curve(y_true  = y_test, 
                                              y_score = y_pred)
    roc_auc = metrics.auc(x = fpr, y = tpr)

    plt.plot(fpr, tpr,
                color     = 'blue',
                linestyle = 'dashed',
                label     = 'Linear SVM (auc = %0.2f)' % (roc_auc))
    
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],
                linestyle = '--',
                color     = 'gray',
                linewidth = 2)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    

def exp_two(X_train, y_train, X_test, y_test, svm):
    wm = svm.coef_

if __name__ == "__main__":
    sys.exit(int(main() or 0))