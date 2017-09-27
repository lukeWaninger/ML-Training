from sklearn.ensemble import RandomForestClassifier
import numpy as np

#show the feature importance using sklearn random forest
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)
forest.fit(x_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

#plot it
#this shows that the color intensity is the most important feature
#when considering which ones to remove during dimension reduction
#-- still be careful of feature correlation
plt.clf()
plt.title('Feature Importances')
plt.bar(range(x_train.shape[1]),
        importances[indices],
        color = 'lightblue',
        align = 'center')
plt.xticks(range(x_train.shape[1]),
           feat_labels[indices], rotation = 90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
"""show the plot"""

#this idea can be used in the random forest pipeline by setting
#the threshold to .15 which will limit the feature decisions to
#the top three features when generating the forest ensemble
x_selected = forest.transform(x_train, threshold = 0.15)
x_selected.shape