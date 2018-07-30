# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 00:23:53 2018

@author: atulr
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd




df= pd.read_csv('./Stockdata/ethi.csv')



X=df["Open"]
Y=df["Close"]




dim= df.shape[0]

print(dim)

per=( .7 *dim)
print(per)
 
# Split the data into training/testing sets
X_train = X[:0,1]
X_test = X[:,1]
 
# Split the targets into training/testing sets
Y_train = Y[:,18]
Y_test = Y[:,18]

regr = linear_model.LinearRegression()
 
# Train the model using the training sets
regr.fit(X_train, Y_train)
 
# Plot outputs
p=regr.predict(X_test)