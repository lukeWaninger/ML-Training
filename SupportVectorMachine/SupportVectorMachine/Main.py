from sklearn.svm                import LinearSVC
import matplotlib.pyplot        as plt
import sklearn.metrics          as metrics
import numpy                    as np
import pandas                   as pd
import os, sys, collections

def main():
    # read in the data
    data = pd.read_hdf("spambase.hdf", header = None)
    X_train, y_train, X_test, y_test = data.values[0][0], data.values[1][0], data.values[2][0], data.values[3][0]

    svm = exp_one(X_train, y_train, X_test, y_test, 'exp_1_v')
    exp_two(X_train, y_train, X_test, y_test, svm, 'exp_2_v')
    exp_three(X_train, y_train, X_test, y_test, 'exp_3_v')

def exp_one(X_train, y_train, X_test, y_test, filename):
    # initialize and fit the SVM
    svm = LinearSVC() #random_state = 1)
    svm.fit_transform(X_train, y_train)

    # predict with the test set
    y_pred = svm.predict(X_test)

    # output accuracy, precision, and recall to text file
    f = open(filename + '.txt', 'a')
    f.write('Accuracy Score: %.3f; Precision: %.3f; Recall %.3f\n' %
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
    return svm
    
def exp_two(X_train, y_train, X_test, y_test, svm, filename):
    # sort the training set feature space based on ||w_i||
    rank = np.argsort(svm.coef_)[0][::-1]
    X_train_sorted = X_train[:, rank]
    X_test_sorted  = X_test[:, rank]

    # show principle components (not part of assignment)
    cov_mat = np.cov(X_train.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)    

    # plot the variance explained ratio to show how the first dimensions
    # account for the variance of the matrix
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse = True)]
    
    acc = []
    # train and record accuracies for models including i features
    for i in range(1, rank.shape[0] - 1, 1):
        svc = LinearSVC()
        X_train_compressed = X_train_sorted[:,:i]
        X_test_compressed  = X_test_sorted[:,:i]
        svc.fit_transform(X_train_compressed, y_train)

        # predict with the test set
        y_pred = svc.predict(X_test_compressed)
        acc.append(metrics.accuracy_score(y_test, y_pred))
    
    # setup plot
    plt.clf()
    plt.grid()
    plt.ylim((np.min(acc), .95))
    plt.xlim((0, 57))
    plt.title('Accuracy per Features Used')
    plt.xlabel('Features Included')
    plt.ylabel('ROC Accuracy Score')

    # plot the accuracy
    plt.plot([i + 1 for i in range(len(acc))], 
             acc,
             color     = 'blue',
             linestyle = 'solid',
             linewidth = 3.0,
             label     = 'Accuracy')

    # plot the feature variance
    min = np.min(acc)
    var_exp_offset = [min for l in range(len(var_exp))]
    plt.bar(range(0, len(var_exp)), var_exp_offset, align = 'center', edgecolor = 'none', color = 'white')
    plt.bar(range(0, len(var_exp)), 
            var_exp, 
            alpha  = 0.3,
            color  = 'blue',
            edgecolor = 'blue',
            align  = 'edge',
            bottom = var_exp_offset, 
            label  = 'Feature Covariance (ranked)')       
    plt.legend(loc = 'upper left',
               ncol     = 1, 
               fontsize = 'small',
               fancybox = True, 
               shadow   = True)

    # print some misc info
    f = open(filename + '.txt', 'a')
    f.write('\n')
    [f.write('%d, ' % rank[k]) for k in range(rank.shape[0])]    
    f.close()

    # save data to file for later use
    plt.savefig(filename)

def exp_three(X_train, y_train, X_test, y_test, filename):
    acc = []
    # train models including the i number of random features
    for i in range(2, X_train.shape[1] - 1):
        # generate random order of features
        order = np.random.permutation(len(X_test[0]))
        X_train_rand = X_train[:, order]
        X_test_rand  = X_test[:, order] 

        svc = LinearSVC()
        X_train_compressed = X_train_rand[:,:i]
        X_test_compressed  = X_test_rand[:,:i]
        svc.fit_transform(X_train_compressed, y_train)

        # predict with the test set
        y_pred = svc.predict(X_test_compressed)
        acc.append(metrics.accuracy_score(y_test, y_pred))
    
    # plot the data
    plt.ylim((np.min(acc), .95))
    plt.plot([i + 1 for i in range(len(acc))], 
             acc,
             color     = 'red',
             linewidth = 3.0,
             linestyle = 'solid',
             label     = 'Random Selection')        

    # calculate feature covariance
    cov_mat = np.cov(X_train.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)    

    # plot the variance explained ratio to show how the first dimensions
    # account for the variance of the matrix
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in eigen_vals]
    var_exp_orderd = [var_exp[k] for k in range(len(var_exp))]
    var_exp_offset = [np.min(acc) for l in range(len(var_exp))]
    plt.bar(range(0, len(var_exp)), var_exp_offset, align = 'center', edgecolor = 'none', color = 'white')
    plt.bar(range(0, len(var_exp)), 
            var_exp_orderd, 
            alpha  = 0.3,
            color  = 'red',
            edgecolor = 'red',
            align  = 'edge',
            bottom = var_exp_offset, 
            label  = 'Feature Covariance (random)')    
    plt.legend(loc = 'upper left',
               ncol = 2, 
               fancybox = True, 
               fontsize = 'small',
               shadow = True)
    plt.savefig(filename + '_all')

    # plot experiment 3 data by itself
    plt.clf()
    plt.ylim((np.min(acc), np.max(acc) + .02))
    plt.xlim((0, 57))
    plt.plot([i + 1 for i in range(len(acc))], 
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
            label  = 'Feature Covariance')
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
    plt.grid()
    plt.savefig(filename)

if __name__ == "__main__":
    sys.exit(int(main() or 0))