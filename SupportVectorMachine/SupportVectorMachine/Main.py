from sklearn.cross_validation   import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.svm                import LinearSVC
import matplotlib.pyplot        as plt
import sklearn.metrics          as metrics
import numpy                    as np
import pandas                   as pd
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
    svm = exp_one(X_train, y_train, X_test, y_test, 'exp_1')
    #exp_two(X_train, y_train, X_test, y_test, svm, 'exp_2')
    #exp_three(X_train, y_train, X_test, y_test, 'exp_3')

def exp_one(X_train, y_train, X_test, y_test, filename):
    # initialize and fit the SVM
    svm = LinearSVC(random_state = 1)
    svm.fit_transform(X_train, y_train)

    # predict with the test set
    y_pred = svm.predict(X_test)
    print('Accuracy Score: %.3f; Precision: %.3f; Recall %.3f' %
          (metrics.accuracy_score(y_test, y_pred),
           metrics.precision_score(y_test, y_pred),
           metrics.recall_score(y_test, y_pred)))

    # get false/true positive rates and threshold from roc_curve
    y_pred = svm.decision_function(X_test)
    fpr, tpr, threshholds = metrics.roc_curve(y_true  = y_test, 
                                              y_score = y_pred)
    roc_auc = metrics.auc(x = fpr, y = tpr)

    # plot the ROC
    plt.clf()
    plt.plot(fpr, tpr,
             color     = 'blue',
             linestyle = 'dashed',
             label     = 'Linear SVM (auc = %0.2f)' % (roc_auc))
    
    plt.title('Receiver Operating Characteristics (ROC)')
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

    # export the figure and data
    plt.savefig(filename)
    pd.to_pickle((svm.coef_, fpr, tpr, threshholds, roc_auc), filename)
    return svm
    
def exp_two(X_train, y_train, X_test, y_test, svm, filename):
    # sort the training set feature space based on ||w_i||
    wm = svm.coef_
    rank = np.argsort(svm.coef_)
    for i in range(X_train.shape[0]):
        X_train[i] = np.array([X_train[i][j] for j in rank[0]])
    for i in range(X_test.shape[0]):
        X_test[i] = np.array([X_test[i][j] for j in rank[0]])
    
    
    acc = []
    for i in range(1, rank.shape[1] - 1, 1):
        svc = LinearSVC()
        X_train_compressed = X_train[:,:i]
        X_test_compressed  = X_test[:,:i]
        svc.fit_transform(X_train_compressed, y_train)

        # predict with the test set
        y_pred = svc.predict(X_test_compressed)
        acc.append(metrics.accuracy_score(y_test, y_pred))
    
    plt.clf()
    plt.plot([i for i in range(len(acc))], 
             acc,
             color     = 'blue',
             linestyle = 'solid',
             label     = 'Accuracy')
    
    plt.title('Accuracy per Features Used [ranked]')
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.xlabel('Features Included')
    plt.ylabel('ROC Accuracy Score')
    
    plt.savefig(filename)
    pd.to_pickle((acc, rank), filename)

def exp_three(X_train, y_train, X_test, y_test, filename):
    order = np.random.permutation(len(X_test[0]))
    for i in range(X_train.shape[0]):
        X_train[i] = np.array([X_train[i][j] for j in order])
    for i in range(X_test.shape[0]):
        X_test[i] = np.array([X_test[i][j] for j in order])
    
    acc = []
    for i in range(1, len(order) - 1, 1):
        svc = LinearSVC()
        X_train_compressed = X_train[:,:i]
        X_test_compressed  = X_test[:,:i]
        svc.fit_transform(X_train_compressed, y_train)

        # predict with the test set
        y_pred = svc.predict(X_test_compressed)
        acc.append(metrics.accuracy_score(y_test, y_pred))
    
    plt.clf()
    plt.plot([i for i in range(len(acc))], 
             acc,
             color     = 'blue',
             linestyle = 'solid',
             label     = 'Accuracy')
    
    plt.title('Accuracy per Features by Feature [random]')
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.xlabel('Features Included')
    plt.ylabel('ROC Accuracy Score')
    
    plt.savefig(filename)
    pd.to_pickle((acc, order), filename)

if __name__ == "__main__":
    sys.exit(int(main() or 0))