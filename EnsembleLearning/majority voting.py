from scipy.misc             import comb
import numpy                as np
import matplotlib.pyplot    as plt
import math

# sum of the ensemble errors by way of binomial mass distribution
def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [comb(n_classifier, k) * error**k * (1 - error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)

ensemble_error(n_classifier = 11, error = 0.25)

# plot the ensemble error vs base error
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier = 11,
                             error = error)
              for error in error_range]
plt.plot(error_range, ens_errors,
         label = 'ensemble error',
         linewidth = 2)
plt.plot(error_range, error_range,
         label = 'base error',
         linestyle = '--',
         linewidth = 2)
plt.xlabel('base error')
plt.ylabel('base/ensemble error')
plt.legend(loc = 'upper left')
plt.grid()
""" show the plot """

# tune the inverse regulation parameter C for Logistic Regression
# and the decision tree depth (using mv_clf.get_params() to find attributes)
from sklearn.grid_search import GridSearchCV

params = { 'decisiontreeclassifier__max_depth': [1,2],
           'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator = mv_clf,
                    param_grid = params,
                    cv = 10,
                    scoring = 'roc_auc')
grid.fit(X_train, y_train)

# print the hyperparameter value combinations and the average
# ROC AUC scores computed via 10-fold cross-validation
for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f+/-%0.2f %r)" % (mean_score, scores.std() / 2, params))

# print the best parameters and accuracy
print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)