# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:37:03 2021

@author: sai vamshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:,-1].values


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

print(regressor.predict([[6.5]]))

x_grid = np.arange(min(X),max(X),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(X,Y,color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('True r bluff')
plt.xlabel('Position label')
plt.ylabel('salary')
plt.show()


