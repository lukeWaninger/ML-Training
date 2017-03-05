import sklearn.metrics   as metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import scipy.stats as stats
import KMeans, sys, itertools

save_location = "C:\\Users\\Luke\\OneDrive\\School\\CS 445 [Machine Learning]\\Homework\\Homework 5 - KMeans Clustering\\content\\"
K        = [30]
restarts = [20]
conv_pt  = [1e-5]

def main():
    # read in the data
    X_train  = pd.read_csv("optdigits.train").values
    X_test   = pd.read_csv("optdigits.test").values
    best_kms = []

    for k in K:
        for r in restarts:
            for cp in conv_pt:
                clfs    = [KMeans.KMeans(X_train, k, cp) for i in range(r)]
                for clf in clfs: clf.fit()
                best_km = clfs[np.argmin([c.avg_mse()] for c in clfs)]
                best_kms.append([best_km, k, r, cp])
                y_pred  = [best_km.pred(xi) for xi in X_test]

                # generate the confusion matrix
                plt.clf()
                cm = metrics.confusion_matrix(X_test[:,-1], y_pred)
                plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Purples)
                plt.xticks(range(10), range(10))
                plt.xticks(range(10), range(10))
                plt.ylabel('True Label', fontsize = 14)
                plt.xlabel('Predicted Label', fontsize = 14)

                # plot confusion numbers
                thresh = cm.max() / 2.
                for j, l in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(l, j, cm[j, l],
                             horizontalalignment= "center",
                             verticalalignment  = "center",
                             color="white" if cm[j, l] > thresh else "black")
                                   
                filename = save_location + str(k) + "_" + str(r)
                plt.savefig(filename, bbox_inches='tight')

                # write metric row: avgMSE, MSS, ACC, K, conv_pt, num_restarts
                f = open("%smetrics.csv" % save_location, 'a')
                f.write('%.3f, %.3f, %.3f, %d, %f, %d\n' %
                       (best_km.avg_mse(),
                        best_km.mss(),
                        metrics.accuracy_score(X_test[:,-1], y_pred),
                        k, cp, r))       
                f.close()

    # filename: best_km_number of clusters used_number of restarts_convergence point.hdf
    pd.DataFrame(best_kms).to_hdf("best_kms.hdf", "hw5")
     
if __name__ == "__main__":
    sys.exit(int(main() or 0))