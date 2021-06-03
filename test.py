# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:27:07 2021

@author: sai vamshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[: ,-1].values

print(X)