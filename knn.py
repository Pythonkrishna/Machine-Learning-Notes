# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:20:46 2021

@author: sai vamshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[: ,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)
print(X_test)


from sklearn.neighbors import KNeighborsClassifier
classfier = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
classfier.fit(X_train,y_train)

print(classfier.predict(sc.transform([[30,87000]])))

y_pred = classfier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))








