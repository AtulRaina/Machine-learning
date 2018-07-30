import pandas as pd 
import numpy as nu 
import matplotlib as mp 


sql='SELECT * FROM [Northwind].[dbo].[Employees]'

df=pd.read_sql(sql,'arainnov\SQLEXPRESS','','','Northwind')
print(df.describe())
df= pd.read_csv('c:/pythonpanda/stockdata/ethi.csv')

print(df.describe())
md= df["Open"]                 
#size=df.shape(0)

x=md[:5]
y=md[3:4]
print(x)
print(y)

df=pd.read_sql("Select * from ")
