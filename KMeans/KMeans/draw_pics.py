import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import sys

def main():
    save_location = "C:\\Users\\Luke\\OneDrive\\School\\CS 445 [Machine Learning]\\Homework\\Homework 5 - KMeans Clustering\\content\\"
    best_kms = pd.read_hdf("best_kms.hdf", "hw5", header=None).values

    for clf in best_kms:
        plt.clf()
        c, K, r, cp = clf[0], clf[1], clf[2], clf[3]   
        #cpstr = str(cp).split('.')[1]    
        plt.set_cmap('bone')
        plt.tick_params(
            axis   = 'both',     # changes apply to the x-axis
            which  = 'both',     # both major and minor ticks are affected
            bottom = 'off',      # ticks along the bottom edge are off
            top    = 'off',      # ticks along the top edge are off
            labelbottom = 'off') # labels along the bottom edge are off

        for m in c.C:
            if m[2] < 0: m[2] = 99
            title = "K-%d Digit-%d" % (K, m[2])
            plt.title(title)
            plt.imshow(m[0].reshape((8,8)))
            #filename = save_location + title + "_" + cpstr
            filename = save_location + title
            plt.savefig(filename)
     
if __name__ == "__main__":
    sys.exit(int(main() or 0))