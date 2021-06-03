# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 14:07:47 2021

@author: sai vamshi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: ,1:-1].values # year of experience
Y = dataset.iloc[:,-1].values


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or bluff')
plt.xlabel('position of values')
plt.ylabel('salary')
plt.show()


plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X,)),color='blue')
plt.title('Truth or bluff')
plt.xlabel('position of values')
plt.ylabel('salary')
plt.show()

print(regressor.predict([[8.5]])) # simple linear
print(lin_reg2.predict(poly_reg.fit_transform([[8.5]]))) # polynominal











