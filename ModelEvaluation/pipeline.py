# start the pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model  import LogisticRegression
from sklearn.pipeline      import Pipeline

pipe_lr = Pipeline([('scl', 
                    StandardScaler()),
                    ('pca', PCA(n_components = 2)),
                    ('clf', LogisticRegression(random_state = 1))])
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

# stratified k-fold
import numpy as np
from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y = y_train,
                        n_folds = 10,
                        random_state = 1)
scores = []

for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(scores)
    print('Fold: %s, class dist.: %s, Acc: %.3f' % 
          (k + 1, np.bincount(y_train[train]), score))

print('CV accuracy: %.3f +/- %.3f' %
      (np.mean(scores),
       np.std(scores)))

# or just use skicits built in k-fold scorer
# n_jobs specifies how many cpu's for parallelism
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator = pipe_lr, 
                         X = X_train,
                         y = y_train,
                         cv = 10,
                         n_jobs = 1)

print('CV Accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' %
      (np.mean(scores),
       np.std(scores)))