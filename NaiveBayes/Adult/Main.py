import pandas as pd
import numpy  as np
import sklearn.preprocessing as preproc
from sklearn import naive_bayes
import sys

def main():
    # load training data
    training_data = pd.read_csv("adult.data", header = None)
    X, y = training_data.iloc[:,:-1].values, training_data.iloc[:,-1].values

    # encode the class labels
    enc = preproc.LabelEncoder()
    enc.fit(y)
    y = enc.transform(y)

    # load test data
    test_data = pd.read_csv("adult.test", header = None)
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:,-1].values
    y_test = [y[:-1] for y in y_test]
    y_test = enc.transform(y_test)

    # find class probabilities
    p0 = np.bincount(y)[0]/len(y)
    p1 = 1 - p0

    # split samples by classes
    c0, c1 = [], []
    for x, y in zip(X, y):
        if y == 0: c0.append(x)
        else: c1.append(x)

    # find probabilities for continuous features
    cont_features = [0,2,4,10,11,12]
    dscrt_features = [1,3,5,6,7,8,9,13]

    # gather conditional probabilities
    def proba_table(feature_space):
        cond_prob = {}

        # loop through each feature
        for i, prop in enumerate(feature_space):
            freq_table = {}

            # loop through each x_{i,j} counting each occurrence
            for val in prop:
                if val not in freq_table:
                    freq_table[val] = 1
                freq_table[val] += 1

            # calculate the probability and add it to the dictionary
            for key in freq_table:
                cond_prob[key] = freq_table[key]/len(prop)

        return cond_prob

    c0_probs, c1_probs = proba_table(np.array(c0)[:,dscrt_features].T), \
                         proba_table(np.array(c1)[:,dscrt_features].T)

    # compute mean and standard deviations for continuous features
    c0_ms = np.array([(np.mean(xi), np.std(xi)) for xi in np.array(c0)[:, cont_features].T])
    c1_ms = np.array([(np.mean(xi), np.std(xi)) for xi in np.array(c1)[:, cont_features].T])

    # define something to make predictions
    def class_prob(sample, ms, class_proba):
        cp = []

        for i, f in enumerate(sample):
            a = 0
            if i in cont_features:
                cp.append((gaussian_pdf(f, ms[a][1], ms[a][0])))
                a += 1
            else:
                cp.append(c0_probs[f])

        for p in cp:
            if p != 0:
                class_proba *= p

        return class_proba

    predictions = [np.argmax([class_prob(sample, c0_ms, p0), class_prob(sample, c1_ms, p1)]) for sample in X_test]

    print("end")

def gaussian_pdf(x,s,m):
    return np.nan_to_num((1/(np.sqrt(2*np.pi)*s)) * np.exp(-((x-m)**2/(2*s**2))))



if __name__ == "__main__":
    sys.exit(int(main() or 0))