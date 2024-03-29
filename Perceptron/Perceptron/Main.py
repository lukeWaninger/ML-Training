import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy  as np
from sklearn.metrics import confusion_matrix
import sys, itertools, winsound, pickle, Perceptron

iw_bounds   = [-.05, .05]
train_size  = 5000
test_size   = 500
eta         = [1e-2, 1e-3, 1e-4]
des_conv_pt = 1e-3

def main():
    # load the data
    try:
        X_train, y_train, X_test, y_test = de_pickle()
    except:
        sys.exit(0)

    # plot accuracies and confusion matrix per eta
    fig, ax = plt.subplots(nrows = len(eta), ncols = 2)
    for n, i in zip(eta, range(len(eta))):
        train_acc, test_acc, loss = [], [], []              

        # get a perceptron ready and train with the given values
        p = Perceptron.Perceptron(y_train, train_size, test_size, iw_bounds)
        train_acc, test_acc, loss = \
            p.learn(X_train, y_train, n, des_conv_pt, X_test, y_test)

        # plot accuracies per epoch
        ax[i][0].set_title('Accuracy per Epoch - Learning Rate: %f' % (n))
        ax[i][0].set_xlabel('Epochs')
        ax[i][0].set_ylabel('Accuracy')
        ax[i][0].plot(train_acc, color = 'green', label = 'Train Acc.')
        ax[i][0].plot(test_acc,  color = 'red',   label = 'Test Acc.')

        # setup the legend
        box = ax[i][0].get_position()
        ax[i][0].set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])
        ax[i][0].legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.22), ncol = 3)

        # generate confusion matrix        
        X = np.insert(X_test, 0, 1, axis = 1)
        y_pred = np.array([p.pre(xi) for xi in X])
        cm = confusion_matrix(y_test, y_pred)

        ax[i][1].imshow(cm, interpolation = 'nearest', cmap = plt.cm.Greens)
        ax[i][1].set_xticks(range(9), range(9))
        ax[i][1].set_yticks(range(9), range(9))
        ax[i][1].set_ylabel('True Label')
        ax[i][1].set_xlabel('Predicted Label')

        # plot text representation of numbers in cell of the conf mat
        thresh = cm.max() / 2.
        for j, k in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax[i][1].text(k, j, cm[j, k],
                     horizontalalignment= "center",
                     verticalalignment  = "center",
                     color="white" if cm[j, k] > thresh else "black")

    plt.tight_layout(pad = 0.2, w_pad = 0.2, h_pad = 0.5)
    winsound.PlaySound('sound.wav', winsound.SND_FILENAME) # notify when you're done
    plt.show()

def load_data(train_size = train_size, test_size = test_size):
    """ Loads the MNIST training and test datasets then stores them
    as a byte array in the running directory
    
    Parameters
    ----------
    train_size : int : range from 0 to 60000. Determines how many
        training samples to load from the csv file
    test_size : int : range from 0 to 10000. Determines how many test
        samples to load from the csv file
    """
    # load the data
    df_train = pd.read_csv('./mnist_train.csv')
    df_test  = pd.read_csv('./mnist_test.csv')

    y_train  = df_train.iloc[:train_size, 0].values
    y_train  = y_train.reshape([y_train.shape[0], 1])
    X_train  = df_train.iloc[:train_size, 1:].values

    y_test   = df_test.iloc[:test_size, 0].values
    y_test   = y_test.reshape([y_test.shape[0], 1])
    X_test   = df_test.iloc[:test_size, 1:].values

    # scale the data from 0, 1
    X_test  = X_test/255
    X_train = X_train/255

    to_pickle = [X_train, y_train, X_test, y_test]
    pickle_me_this_batman(to_pickle)

def pickle_me_this_batman(to_pickle):
    """ Stores the provided object as a byte array to the current
    directory as 'data.picke'

    Parameters
    -----------
    to_pickle : object : writes it as a byte array

    Returns
    -----------
    True or False as a success flag
    """
    try:
        with open('data.pickle', 'wb') as f:
            pickle.dump(to_pickle, f, pickle.HIGHEST_PROTOCOL)
    except:
        return False
    return True

def de_pickle():
    """ Reads the byte array 'data.pickle' from the current directory

    Returns
    --------
    X_train, y_train, X_test, y_test as they were stored
    """
    de_pickled = []

    try:
        with open('data.pickle', 'rb') as f:
            de_pickled = pickle.load(f)
    except:
        load_data()
        return

    return de_pickled[0], de_pickled[1], de_pickled[2], de_pickled[3]

if __name__ == "__main__":
    sys.exit(int(main() or 0))