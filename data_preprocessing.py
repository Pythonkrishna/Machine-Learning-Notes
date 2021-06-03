# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:10:13 2021

@author: sai vamshi
"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#print(X)
#print(Y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)

# encoding the data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
print(X)

