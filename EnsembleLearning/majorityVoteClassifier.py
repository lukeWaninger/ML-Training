from sklearn.base           import BaseEstimator
from sklearn.base           import ClassifierMixin
from sklearn.preprocessing  import LabelEncoder
from sklearn.externals      import six
from sklearn.base           import clone
from sklearn.pipeline       import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers: array-like, shape = [n_classifiers]
        different classifiers for the ensemble

    vote: str, {'class-label', 'probability'}
        default: 'class-label'
        If 'class-label' the prediction is based on the argmax
        of class labels. Else if 'probability', the argmax of 
        the sum of probabilities is used to predict the class 
        label. (recommended for calibrated classifiers).

    weights: array-like, shape = [n_classifiers]
        optional, default: none
        If a list of 'int' or 'float' values are provided, the
        classifiers are weighted by importance; uses uniform weights
        if weights = none
    """
    def __init__(self, classifiers, vote = 'classlabel', weights = None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for
                                  key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.
        
        Parameters
        -----------
        X: {array-like, sparse matrix},
           shape = [n_samples, n_features]
           matrix of training samples
           
        y: array-like, shape = [n_samples]
           vector of target class labels.
             
        Returns
        ---------
        self: object
        """

        # use LabelEncoder to ensure class labels start
        # with 0, which is important for np.argmax
        # call self.predict

        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """ predict class labels for X.

        Parameters
        -----------
        X: {array-like, sparse matrix},
            shape = [n_samples, n_features]
            matrix of training samples

        Returns
        ----------
        maj_vote: array-like, shape = [n_samples]
            predicted class labels.
        """

        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis = 1)
        else:
            # 'classlabel' vote
            # collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda x:
                    np.argmax(np.bincount(x, weights = self.weights)),
                    axis = 1,
                    arr = predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)

        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        -----------
        X: {array-like, sparse matrix},
        shape = [n_samples, n_features]
        training vectors, where n_samples is the number of samples
        and n_features is the number of features

        returns
        ----------
        avg_proba : array-like,
            shape = [n_samples, n_classes]
            weighted average probability for each class per samples
        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis = 0, weights = self.weights)
        return avg_proba

    def get_params(self, deep = True):
        """ get classifier parameter names for GridSearch """

        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep = False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(
                    step.get_params(deep = True)):
                    out['%s__%s' % (name, key)] = value
        return out