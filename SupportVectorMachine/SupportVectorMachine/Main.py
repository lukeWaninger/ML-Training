from sklearn.cross_validation   import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.preprocessing      import MinMaxScaler
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
    X_test  = stdsc.transform(X_test)

    # show class distribution
    #print(collections.Counter(X_train))
    #print(collections.Counter(X_test))
    svm = exp_one(X_train, y_train, X_test, y_test, 'exp_1')
    exp_two(X_train, y_train, X_test, y_test, svm, 'exp_2')
    exp_three(X_train, y_train, X_test, y_test, 'exp_3')

def exp_one(X_train, y_train, X_test, y_test, filename):
    # initialize and fit the SVM
    svm = LinearSVC(random_state = 1)
    svm.fit_transform(X_train, y_train)

    # predict with the test set
    y_pred = svm.predict(X_test)

    # output accuracy, precision, and recall to text file
    f = open(filename + '.txt', 'w')
    f.write('Accuracy Score: %.3f; Precision: %.3f; Recall %.3f' %
          (metrics.accuracy_score(y_test, y_pred),
           metrics.precision_score(y_test, y_pred),
           metrics.recall_score(y_test, y_pred)))
    f.close()

    # get false/true positive rates and threshold from roc_curve
    y_pred_dec = svm.decision_function(X_test)
    fpr, tpr, threshholds = metrics.roc_curve(y_true  = y_test, y_score = y_pred_dec)
    roc_auc = metrics.auc(x = fpr, y = tpr)

    # plot the ROC
    plt.clf()
    plt.plot(fpr, tpr,
             color     = 'blue',
             linestyle = 'dashed',
             label     = 'Linear SVM (AUC = %0.2f)' % (roc_auc))
    
    # setup the figure
    plt.grid()
    plt.title('Receiver Operating Characteristics (ROC)')
    plt.legend(loc = 'lower right',
               fancybox = True, 
               fontsize = 'small',
               shadow = True,)
    plt.plot([0, 1], [0, 1],
             linestyle = '--',
             color     = 'gray',
             linewidth = 2)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # export the figure and data
    plt.savefig(filename)
    pd.to_pickle((svm.coef_, fpr, tpr, threshholds, roc_auc), filename)
    return svm
    
def exp_two(X_train, y_train, X_test, y_test, svm, filename):
    # sort the training set feature space based on ||w_i||
    wm = svm.coef_
    rank = np.argsort(wm)[0][::-1]
    for i in range(X_train.shape[0]):
        X_train[i] = np.array([X_train[i][j] for j in rank])
    for i in range(X_test.shape[0]):
        X_test[i] = np.array([X_test[i][j] for j in rank])

    # show principle components (not part of assignment)
    cov_mat = np.cov(X_train.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)    

    # plot the variance explained ratio to show how the first dimensions
    # account for the variance of the matrix
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse = True)]
    """
    #cum_var_exp = np.cumsum(var_exp)

    #sort the eigenvectors based on their magnitudes
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i], i) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)

    # next collect the eigenvectors that correspond to the two
    # largest values (60% of the variance). The number of chosen
    # vectors will be a trade off between computational efficiency
    # and classifier performance
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                   eigen_pairs[1][1][:, np.newaxis]))
    """
    
    acc = []
    # train and record accuracies for models including i features
    for i in range(1, rank.shape[0] - 1, 1):
        svc = LinearSVC()
        X_train_compressed = X_train[:,:i]
        X_test_compressed  = X_test[:,:i]
        svc.fit_transform(X_train_compressed, y_train)

        # predict with the test set
        y_pred = svc.predict(X_test_compressed)
        acc.append(metrics.accuracy_score(y_test, y_pred))
    
    # setup plot
    plt.clf()
    plt.grid()
    plt.ylim((np.min(acc), np.max(acc) + .02))
    plt.xlim((0, 57))
    plt.title('Accuracy per Features Used')
    plt.xlabel('Features Included')
    plt.ylabel('ROC Accuracy Score')

    # plot the accuracy
    plt.plot([i for i in range(len(acc))], 
             acc,
             color     = 'blue',
             linestyle = 'solid',
             linewidth = 3.0,
             label     = 'Accuracy')

    # plot the variance by feature
    min = np.min(acc)
    var_exp_offset = [min for l in range(len(var_exp))]
    plt.bar(range(1, len(var_exp) + 1), var_exp_offset, align = 'center', edgecolor = 'none', color = 'white')
    plt.bar(range(0, len(var_exp)), 
            var_exp, 
            alpha  = 0.3,
            color  = 'blue',
            edgecolor = 'blue',
            align  = 'edge',
            bottom = var_exp_offset, 
            label  = 'Feature Variance')       
    plt.legend(loc = 'upper left',
               ncol     = 1, 
               fontsize = 'small',
               fancybox = True, 
               shadow   = True)

    # print some misc info
    f = open(filename + '.txt', 'w')
    [f.write('%d, ' % rank[k]) for k in range(rank.shape[0])]
    #f.write('\n\nEigen Pairs\n-------------\n%s\n' % [eigen_pairs[i][2] for i in range(0, len(eigen_pairs))])
    #f.write('\nEigenvalues\n-------------\n%s\n' % sorted(eigen_vals))
    #f.write('\nEigenvectors\n-------------\n%s\n' % eigen_vecs)
    #f.write('\nMatrix W:\n------------\n%s' % w)
    f.close()
    
    # save data to file for later use
    plt.savefig(filename)
    pd.to_pickle((acc, rank), filename)

def exp_three(X_train, y_train, X_test, y_test, filename):
    # generate random order of features
    order = np.random.permutation(len(X_test[0]))
    for i in range(X_train.shape[0]):
        X_train[i] = np.array([X_train[i][j] for j in order])
    for i in range(X_test.shape[0]):
        X_test[i] = np.array([X_test[i][j] for j in order])
    
    acc = []
    # train models including the i number of random features
    for i in range(1, len(order) - 1, 1):
        svc = LinearSVC()
        X_train_compressed = X_train[:,:i]
        X_test_compressed  = X_test[:,:i]
        svc.fit_transform(X_train_compressed, y_train)

        # predict with the test set
        y_pred = svc.predict(X_test_compressed)
        acc.append(metrics.accuracy_score(y_test, y_pred))
    
    # plot the data
    plt.plot([i for i in range(len(acc))], 
             acc,
             color     = 'red',
             linewidth = 3.0,
             linestyle = 'solid',
             label     = 'Random Selection')        

    cov_mat = np.cov(X_train.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)    

    # plot the variance explained ratio to show how the first dimensions
    # account for the variance of the matrix
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in eigen_vals]
    var_exp_orderd = [var_exp[k] for k in range(len(var_exp))]
    var_exp_offset = [np.min(acc) for l in range(len(var_exp))]
    plt.bar(range(1, len(var_exp) + 1), var_exp_offset, align = 'center', edgecolor = 'none', color = 'white')
    plt.bar(range(0, len(var_exp)), 
            var_exp_orderd, 
            alpha  = 0.3,
            color  = 'red',
            edgecolor = 'red',
            align  = 'edge',
            bottom = var_exp_offset, 
            label  = 'Feature Variance (random order)')    
    plt.legend(loc = 'upper right',
               ncol = 2, 
               fancybox = True, 
               fontsize = 'small',
               shadow = True)
    plt.savefig(filename + '_all')

    plt.clf()
    plt.ylim((np.min(acc), np.max(acc) + .02))
    plt.xlim((0, 57))
    plt.plot([i for i in range(len(acc))], 
             acc,
             color     = 'red',
             linewidth = 3.0,
             linestyle = 'solid',
             label     = 'Accuracy')
    plt.title('Accuracy per Features Used')
    plt.bar(range(1, len(var_exp) + 1), var_exp_offset, align = 'center', edgecolor = 'none', color = 'white')
    plt.bar(range(0, len(var_exp)), 
            var_exp_orderd, 
            alpha  = 0.3,
            color  = 'red',
            edgecolor = 'red',
            align  = 'edge',
            bottom = var_exp_offset, 
            label  = 'Feature Variance')
    plt.legend(loc = 'upper left',
               ncol = 2, 
               fancybox = True, 
               fontsize = 'small',
               shadow = True)
    plt.xlabel('Features Included')
    plt.ylabel('ROC Accuracy Score')
    plt.legend(loc = 'upper left', 
            ncol = 1, 
            fontsize = 'small', 
            fancybox = True, 
            shadow = True) 
    plt.savefig(filename + '_all')
    plt.grid()

    # save to file
    plt.savefig(filename)
    pd.to_pickle((acc, order), filename)

if __name__ == "__main__":
    sys.exit(int(main() or 0))