# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 19:36:02 2018

@author: atulr
"""

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

url = "C:/PythonPanda/stockdata/ethi.csv"
#names = ['Open', 'High', 'Low', 'Close']
names = ['Open','Close']
dataset = pandas.read_csv(url, names=names)

dataset= dataset.dropna()

dataset= dataset.convert_objects(convert_numeric=True)
#>
# shape
#print(dataset.shape)



# head
'''
print(dataset.head(20))


print(dataset.describe())

'''


# class distribution
# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#dataset.hist()
#plt.show()
cleaner=dataset.dropna()
array = cleaner.values

X = array[:,0:1]
Y = array[:,1]
validation_size = .20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

X_train=X_train[~np.isnan(X_train).any(axis=1)]

# Spot Check Algorithms

seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
    
    
    

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()





# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print('KNeighborsClassifier Accuracy Score')
print(accuracy_score(Y_validation, predictions))
print('KNeighborsClassifier Confusion Matrix')
print(confusion_matrix(Y_validation, predictions))
print('classification_report')
#print(classification_report(Y_validation, predictions))


knn = LogisticRegression()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print('LogisticRegression Accuracy Score')
print(accuracy_score(Y_validation, predictions))
print('LogisticRegression Confusion Matrix')
print(confusion_matrix(Y_validation, predictions))
#print('LogisticRegression classification_report')



print(classification_report(Y_validation, predictions))
print('-------------------------------------------------------------')
# Make predictions on validation dataset
