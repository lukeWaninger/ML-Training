from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#use Sequential Backward Selection (SBS) to reduce dimensionality
knn = KNeighborsClassifier(n_neighbors = 2)
sbs = SBS(knn, k_features = 1)
sbs.fit(x_train_std, y_train)

#plot the classification accuracy
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker = 'o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of Features')
plt.grid()
plt.show()