import sklearn.metrics   as metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import KMeans, sys

def main():
    # read in the data
    X_train = pd.read_csv("optdigits.train").values
    X_test  = pd.read_csv("optdigits.test").values

    # experiment one
    clfs    = [KMeans.KMeans(X_train, 10) for i in range(5)]
    for clf in clfs: clf.fit()
    best_km = clfs[np.argmin([c.mss()] for c in clfs)]
    y_pred  = [best_km.pred(xi) for xi in X_test]

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
     
if __name__ == "__main__":
    sys.exit(int(main() or 0))