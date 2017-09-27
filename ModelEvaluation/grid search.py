from sklearn.grid_search        import GridSearchCV
from sklearn.svm                import SVC
from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import StandardScaler
from sklearn.cross_validation   import cross_val_score
from sklearn.tree               import DecisionTreeClassifier
import numpy as np

# brute force search for best parameter values
pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(random_state = 1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid  = [{'clf__C'     : param_range,
                'clf__kernel': ['linear']},
               {'clf__C'     : param_range,
                'clf__gamma' : param_range,
                'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator  = pipe_svc,
                  param_grid = param_grid,
                  scoring    = 'accuracy',
                  cv         = 10,
                  n_jobs     = -1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))

# a better approach is scikit randomized search
gs = GridSearchCV(estimator  = pipe_svc,
                  param_grid = param_grid,
                  scoring    = 'accuracy',
                  cv         = 2,
                  n_jobs     = -1)
scores = cross_val_score(gs, 
                         X_train, 
                         y_train,
                         scoring = 'accuracy',
                         cv = 5)
print('CV accuracy: %.3f +/- %.3f' % (
    np.mean(scores), np.std(scores)))

gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 0),
                  param_grid = [
                      {'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring = 'accuracy',
                  cv = 5)
scores = cross_val_score(gs,
                         X_train,
                         y_train,
                         scoring = 'accuracy',
                         cv = 2)
print('CV accuracy: %.3f +/- %.3f' % (
    np.mean(scores),
    np.std(scores)))

