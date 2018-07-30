# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 22:40:06 2018

@author: atulr
"""



# figure out the next value of the stock
# will it rise or fall 
# based on the past stock price predict the stock will rise or fall in accuracy
#70

#Using machine learning predict a stock will rise or fall based on the historical 
# trend data with a accuracy of 70 percentage 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename='C:\PythonPanda\data/pima-data.csv'

df=pd.read_csv('C:\PythonPanda\data/pima-data.csv')

print(df.shape)
print(df.head(5))