# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 00:05:38 2018

@author: atulr
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# Load the stock dataset
stock = pd.read_csv('./Stockdata/MVS.csv')

stock=stock.dropna()
# Open hig low 
#stock_X = stock[['Open', 'High', 'Low']]
#Based on One 
stock_X = stock[['Open', 'High', 'Low']]
stock_y = stock[ 'Close']
# Split the data into training/testing sets
stock_X_train = stock_X[:-20]
stock_X_test = stock_X[-20:]

# Split the targets into training/testing sets
stock_y_train =stock_y[:-20]
stock_y_test = stock_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(stock_X_train, stock_y_train)

# Make predictions using the testing set
stock_y_pred = regr.predict(stock_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(stock_y_test, stock_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(stock_y_test, stock_y_pred))
#7.620000,7.630000,7.550000,7.570000,7.447722,228953
#6.260000,6.260000,6.240000,6.250000
predictionData = np.array([ [20.460,20.460,20.360]])
predictionData1 = np.array([ [6.520,6.44,6.500]])
print((regr.predict(predictionData))) 
print((regr.predict(predictionData1))) 
# Plot outputs

#plt.scatter(stock_X_test, stock_y_test,  color='black')
#plt.plot(stock_X_test, stock_y_pred, color='blue', linewidth=3)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()
#```