import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                 header = None,
                 sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
              'RM', 'AGE', 'DIS', 'RAD', 'TAX',
              'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

# visualize the data
import matplotlib.pyplot plt
import seaborn as sns

sns.set(style = 'whitegrid', context = 'notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size = 2.5)
""" show the plot """

# create a coorelation matrix and plot as a heat map
import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm,
                 cbar   = True,
                 annot  = True,
                 square = True,
                 fmt    = '.2f',
                 annot_kws   = { 'size': 15},
                 yticklabels = cols,
                 xticklabels = cols)
""" show the plot """

import LinearRegressionGD
X = df[['RM']].values
y = df['MEDV'].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# plot the lgd vs epochs to show adaline convergence
import matplotlib.pyplot as plt
plt.plot(range(1, lr.n_iter + 1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
""" show the plot """