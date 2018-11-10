#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 22:50:30 2018

@author: nishant
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines

# Importing the dataset
dataset = pd.read_csv('hour.csv')
X = dataset.iloc[:, 2:14].values
y = dataset.iloc[:, 16].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
yl=np.log(y_train)

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3,5,7])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, yl)

# Predicting the Test set results
y_predl = regressor.predict(X_test)
y_pred = np.exp(y_predl)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#vis
def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

# Visualising the Test set results
plt.scatter(y_test, y_pred, color = 'red',  alpha=0.04)
plt.title('qq of test set')
plt.xlabel('y')
plt.ylabel('y_pred')
newline([0,0],[6,6])
plt.axhline(y=0, color='k')
plt.show()

#correlation mat
plt.matshow(dataset.corr())

#removing outliers
sd=dataset["cnt"].std()
mean=dataset["cnt"].mean()
dataset1=dataset.loc[dataset["cnt"]<=(mean+3*sd)] 
print(dataset1)
X = dataset1.iloc[:, 2:14].values
y = dataset1.iloc[:, 16].values
