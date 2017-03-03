import sklearn.metrics   as metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import scipy.stats as stats
import KMeans, sys, itertools

K        = 10
restarts = 1
conv_pt  = .001

def main():
    # read in the data
    X_train = pd.read_csv("optdigits.train").values
    X_test  = pd.read_csv("optdigits.test").values

    # experiment one
    clfs    = [KMeans.KMeans(X_train, K, conv_pt) for i in range(restarts)]
    for clf in clfs: clf.fit()
    best_km = clfs[np.argmin([c.avg_mse()] for c in clfs)]
    y_pred  = [best_km.pred(xi) for xi in X_test]

    # generate the confusion matrix
    cm = metrics.confusion_matrix(X_test[:,-1], y_pred)
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Purples)
    plt.xticks(range(10), range(10))
    plt.xticks(range(10), range(10))
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

    # print the metrics
    f = open("metrics" + '.txt', 'a')
    f.write('avgMSE: %.3f; MSS: %.3f, ACC: %.3f - [K: %d, conv_pt: %d, num_restarts: %d]\n' %
           (best_km.avg_mse(),
            best_km.mss(),
            metrics.accuracy_score(X_test[:,-1], y_pred),
            K, conv_pt, restarts))       
    f.close()

    plt.set_cmap('bone')
    pd.DataFrame(best_km.C).to_hdf("best_km.hdf", "hw5")
     
if __name__ == "__main__":
    sys.exit(int(main() or 0))