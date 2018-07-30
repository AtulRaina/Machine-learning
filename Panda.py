# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 22:23:06 2018

@author: atulr
"""


import pandas as pd
import matplotlib as pl



df=pd.read_csv('./Stockdata/ethi.csv')

#print (df.info)


#(df["Open"]-df["Close"]).plot()

''''
print((df["Adj Close"]-df["Open"]).mean())


df["Movement"]=df["Close"]-df["Open"]



df["Deal"]=df["Movement"] >
print(df.describe())

print(df["Deal"].describe())
'''
df["Change"]=df["Open"]- df["Open"].shift()
print(df["Change"].describe())
df["Change"].describe()
df["Growth"]=df["Change"]>0
print(df["Growth"].describe())
df[["Change","Open"]].plot()

df["Change"].plot()

        

