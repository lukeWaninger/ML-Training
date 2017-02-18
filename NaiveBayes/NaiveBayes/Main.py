import sklearn.metrics   as metrics
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import os, sys, itertools

def main():
    # read in the data
    data = pd.read_hdf("spambase.hdf", header = None)
    X_train, y_train, X_test, y_test = data.values[0][0], data.values[1][0], data.values[2][0], data.values[3][0]

    # find class probabilities
    p_0 = np.bincount(y_train)[0]/y_train.shape[0]
    p_1 = 1 - p_0

    # split samples by class
    c1_train, c2_train = [], []
    for x, y in zip(X_train, y_train):
        if (y == 1): c1_train.append(x)
        else:        c2_train.append(x)
    c1_train = np.nan_to_num(np.array(c1_train))
    c2_train = np.nan_to_num(np.array(c2_train))

    # standard deviation and mean for each feature given a specific class
    c1_sm = np.array([(np.std(xi), np.mean(xi)) for xi in c1_train.T])
    c2_sm = np.array([(np.std(xi), np.mean(xi)) for xi in c2_train.T])

    # make some predictions
    y_pred = []
    for sample in X_test:
        pc0 = np.log(p_0) * np.sum([gaussian_pdf(xi, s, m) for xi, s, m in zip(sample, c1_sm[:,0], c1_sm[:,1])])
        pc1 = np.log(p_1) * np.sum([gaussian_pdf(xi, s, m) for xi, s, m in zip(sample, c2_sm[:,0], c2_sm[:,1])])
        y_pred.append(np.argmax([pc0, pc1]))
       
    # generate the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Greens)
    plt.xticks([0, 1], ["Spam", "~Spam"])
    plt.yticks([0, 1], ["Spam", "~Spam"])
    plt.ylabel('True Label', fontsize = 14)
    plt.xlabel('Predicted Label', fontsize = 14)

    # plot confusion numbers
    thresh = cm.max() / 2.
    for j, k in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(k, j, cm[j, k],
                    horizontalalignment= "center",
                    verticalalignment  = "center",
                    color="white" if cm[j, k] > thresh else "black")
    plt.savefig("cm", bbox_inches='tight')
    
    # print the accuracy
    f = open("metrics" + '.txt', 'a')
    f.write('Accuracy Score: %.3f; Precision: %.3f; Recall %.3f\n' %
           (metrics.accuracy_score(y_test, y_pred),
            metrics.precision_score(y_test, y_pred),
            metrics.recall_score(y_test, y_pred)))        
    f.close()

def gaussian_pdf(x,s,m):
    return np.nan_to_num(np.log((1/(np.sqrt(2*np.pi)*s)) * np.exp(-((x-m)**2/(2*s**2)))))


if __name__ == "__main__":
    sys.exit(int(main() or 0))