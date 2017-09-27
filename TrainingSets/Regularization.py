#For L2 regularization, increase the penalty
#for large individual weights
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training Accuracy:', lr.score(X_train_std, y_train))
print('Test Accuracy:', lr.score(X_test_std, y_test))

#to see the intercepts when considering L1 regularization
#the three values produced are the y-intercepts of class's vs each other
lr.intercept_

#to show their coefficients [the weight array]
lr.coef_

#plot the regularization path - or the feature
#coefficients for different regularization strengths
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue',    'green',      'red', 'cyan', 
          'magenta', 'yellow',     'black', 
          'pink',    'lightgreen', 'lightblue', 
          'gray',    'indigo',     'orange'] 
weights, params = [], [] 

for c in np.arange(-4, 6): 
    lr = LogisticRegression(penalty = 'l1', 
                            C = 10** c, 
                            random_state = 0) 
    lr.fit(X_train_std, y_train) 
    weights.append(lr.coef_[ 1]) 
    params.append(10** c) 

weights = np.array(weights) 

for column, color in zip(range( weights.shape[ 1]), colors): 
    plt.plot(params, weights[:, column], label = df_wine.columns[ column + 1], 
             color = color)

plt.axhline( 0, color = 'black', linestyle ='--', linewidth = 3)
plt.xlim([ 10**(-5), 10** 5]) 
plt.ylabel('weight coefficient') 
plt.xlabel('C')
plt.xscale('log') 
plt.legend(loc ='upper left')
ax.legend( loc ='upper center', bbox_to_anchor =( 1.38, 1.03), ncol = 1, fancybox = True) 
plt.show()