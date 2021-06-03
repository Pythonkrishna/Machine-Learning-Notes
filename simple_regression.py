# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:21:04 2021

@author: sai vamshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
r = dataset.isna().sum()

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# spliting dataset into training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
# Training the simple linear regression model on the test set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#graph
plt.scatter(x_train, y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Experience[Training set]')
plt.xlabel('year of experience')
plt.ylabel('salary')
plt.show()


plt.scatter(x_test, y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Experience[Training set]')
plt.xlabel('year of experience')
plt.ylabel('salary')
plt.show()


