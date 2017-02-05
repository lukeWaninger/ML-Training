import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy  as np
from sklearn.metrics  import confusion_matrix
import sys, itertools, winsound, pickle, NeuralNet

train_size = 15000
test_size  = 1000
eta        = 1e-1

def main():
    # load the data
    try:
        data = de_pickle('data')
        X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
    except:
        sys.exit(0)

    file_names = ['exp_1', 'exp_2', 'exp_3']    

    # run experiments
    experiment_one(X_train, y_train, X_test, y_test, file_names[0])
    experiment_two(X_train, y_train, X_test, y_test, file_names[1])
    experiment_three(X_train, y_train, X_test, y_test, file_names[2])
    
    # plot the data
    for name in file_names:
        plot_data(de_pickle(name), name)

    winsound.PlaySound('sound.wav', winsound.SND_FILENAME) # notify when you're done
    
def experiment_one(X_train, y_train, X_test, y_test, filename):
    alpha = 0.9
    hidden_units = [10, 20, 30]
    experiment_one_data = []

    # learn and gather resulting data
    for n, i in zip(hidden_units, range(len(hidden_units))):     
        # initialize and train the model  
        nn = NeuralNet.NeuralNet(X_train.shape[1], n, eta, alpha)
        train_acc, test_acc = nn.learn(X_train, y_train, X_test, y_test)
        
        # generate confusion matrix data    
        y_pred = np.array([nn.predict(xi) for xi in X_test])     
        
        # append to experiment data
        experiment_one_data.append((train_acc, test_acc, y_test, y_pred, 'Hidden Units: %d' % n, nn))  

    # pickle and plot
    pickle_me_this_batman(experiment_one_data, filename)

def experiment_two(X_train, y_train, X_test, y_test, filename):
    experiment_two_data = []
    hidden_units = 35
    momentum     = [0, 0.25, 0.5]    

    # learn and gather resulting data
    for alpha, i in zip(momentum, range(len(momentum))):
        # initialize and train the model    
        nn = NeuralNet.NeuralNet(X_train.shape[1], hidden_units, eta, alpha)
        train_acc, test_acc = nn.learn(X_train, y_train, X_test, y_test)
        
        # generate confusion matrix        
        y_pred = np.array([nn.predict(xi) for xi in X_test])
       
        # add to resulting data set
        experiment_two_data.append((train_acc, test_acc, y_test, y_pred, 'Momentum: %d' % alpha, nn))

    # pickle and plot
    pickle_me_this_batman(experiment_two_data, filename)

def experiment_three(X_train, y_train, X_test, y_test, filename):
    experiment_three_data = []
    hidden_units = 35
    momentum     = 0.9
    
    # generate random permutation
    r = np.random.permutation(len(X_train))
    X_train, y_train = X_train[r], y_train[r]
    
    # split the training sets into half and quarter sizes
    half    = np.ceil(len(X_train) * .5)
    quarter = np.ceil(len(X_train) * .25)
    X_train_half, y_train_half = X_train[:half,:], y_train[:half,:]
    X_train_quarter, y_train_quarter = X_train[half:half + quarter,:], y_train[half:half + quarter,:]

    X = [X_train_half, X_train_quarter]
    y = [y_train_half, y_train_quarter]

    # learning and gather resulting data
    for x_set, y_set, i in zip(X, y, range(len(X))):
        # initialize and train the model      
        nn = NeuralNet.NeuralNet(x_set.shape[1], hidden_units, eta, momentum)
        train_acc, test_acc = nn.learn(X_train, y_train, X_test, y_test)        

        # generate data for confusion matrix        
        y_pred = np.array([nn.predict(xi) for xi in X_test])
        
        # add to resulting data set
        experiment_three_data.append((train_acc, test_acc, y_test, y_pred, 'Training Samples: %d' % len(x_set), nn))

    # pickle and plot the data
    pickle_me_this_batman(experiment_three_data, filename)

def plot_data(experiment_data, filename):
    # plot accuracies and confusion matrix per eta
    fig, ax = plt.subplots(nrows     = len(experiment_data), 
                           ncols     = 2,
                           figsize   = (25, 25),
                           dpi       = 80, 
                           facecolor = 'w', 
                           edgecolor = 'w')
    # plot accuracies per epoch
    color_map = ['blue', 'green', 'red']
    for i, exp in zip((range(len(experiment_data))), experiment_data):
        ax[0][0].set_title('Accuracy per Epoch', fontsize = 16)
        ax[0][0].set_xlabel('Epochs', fontsize = 14)
        ax[0][0].set_ylabel('Accuracy', fontsize = 14)
        ax[0][0].plot(exp[0], color = color_map[i%3], linestyle = 'solid',  label = 'Train ' + exp[4])
        ax[0][0].plot(exp[1], color = color_map[i%3], linestyle = 'dashed', label = 'Test ' + exp[4])

        # setup the legend
        box = ax[0][0].get_position()
        ax[0][0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax[0][0].legend(loc = 'lower center', ncol = 3)
        
        # generate the confusion matrix
        cm = confusion_matrix(exp[2], exp[3])
        ax[i][0].set_title(exp[4], fontsize = 16)
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

        plt.subplots_adjust(left   = 0.2, 
                            right  = 0.6, 
                            bottom = 0.3, 
                            top    = 0.7,
                            wspace = 0.3,
                            hspace = 0.3) 
    plt.savefig(filename, bbox_inches='tight')
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
    pickle_me_this_batman(to_pickle, 'data')

def pickle_me_this_batman(to_pickle, filename):
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
        with open(filename + '.pickle', 'wb') as f:
            pickle.dump(to_pickle, f, pickle.HIGHEST_PROTOCOL)
    except:
        return False
    return True

def de_pickle(filename):
    """ Reads the byte array 'data.pickle' from the current directory

    Returns
    --------
    X_train, y_train, X_test, y_test as they were stored
    """
    de_pickled = []

    try:
        with open(filename + '.pickle', 'rb') as f:
            de_pickled = pickle.load(f)
    except:
        load_data()
        return

    return de_pickled

if __name__ == "__main__":
    sys.exit(int(main() or 0))