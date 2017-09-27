import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
df

"""To show which collumns have how many NaN fields"""
df.isnull().sum()

"""to show the array"""
df.values

"""to remove any row with a NaN"""
df.dropna()

"""to remove columns with a NaN"""
df.dropna(axis=1)

"""mean imputation is a common method to replace the empty
values with the mean of the feature column"""
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data