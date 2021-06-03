# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:04:09 2021

@author: sai vamshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

print(X)
print('==========================')

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])
print('============================')
print(X)