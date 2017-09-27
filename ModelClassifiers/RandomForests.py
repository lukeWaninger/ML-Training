from sklearn.ensemble import RandomForestClassifier

"""build the random forest. note that n_jobs defines how many
processor cores to use which provides parallelism"""
forest = RandomForestClassifier(criterion='entropy', 
                                n_estimators=10, 
                                random_state = 1, 
                                n_jobs = 2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()