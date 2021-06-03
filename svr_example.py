# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:24:30 2021

@author: sai vamshi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[: ,-1].values

print(X)
#print(Y)
Y = Y.reshape(len(Y),1)
print(Y)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

x = sc_x.fit_transform(X)
y = sc_y.fit_transform(Y)
print(x)
print(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color='red')
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x)),color='blue')
plt.title('True r bluff')
plt.xlabel('Position level')
plt.ylabel('salaries')
plt.show()




















