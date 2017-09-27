import pandas as pd
df = pd.DataFrame(pd.DataFrame([ 
    [' green', 'M', 10.1, 'class1'], 
    [' red', 'L', 13.5, 'class2'], 
    [' blue', 'XL', 15.3, 'class1']]))
df.columns = ['color','size','price','classlabel']
df 

"""map the nominal features to some form of ordering"""
size_mapping = { 'XL':3,
                 'L' :2,
                 'M' :1}
df['size'] = df['size'].map(size_mapping)
df

"""to map back to the previous field values"""
inv_size_mapping = {v: k for k, v in size_mapping.items() }
df['size'] = df['size'].map(inv_size_mapping)

"""convert class labels to an integer enumeration"""
import numpy as np
class_mapping = {label:idx for idx, label in
                 enumerate(np.unique(df['classlabel']))}
class_mapping

"""then transform the class labels into integers"""
df['classlabel'] = df['classlabel'].map(class_mapping)
df

"""or you can use the LabelEncoder provided in the scikit library"""
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

X = df[['color', 'size', 'price']].values 
color_le = LabelEncoder() 
X[:, 0] = color_le.fit_transform( X[:, 0]) 
X

"""casting nominal values into binary columns using two methods
the first uses sklearn's preprocessing one hot encoder"""
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()

"""the second method uses pandas"""
pd.get_dummies(df[['price','color','size']])