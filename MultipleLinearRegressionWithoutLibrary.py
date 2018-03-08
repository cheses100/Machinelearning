#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 17:08:45 2018

@author: danielmizrahi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
temp = np.ones((X.shape[0], X.shape[1] + 1))
temp[:,1:] = X
X = temp

thetas = []
thetas = np.ones(X.shape[1])

alpha = .00000000001
yhat = np.ones(Y.shape)

for i in range(0,1000):
    for j in range(0, X.shape[0]):
        for k in range(0, X.shape[1]):

            thetas[k] = thetas[k] - alpha*(thetas.dot(X[j,:]) - Y[j])*X[j,:][k]
       

deltas = np.ones(Y.shape)
delta_percent= np.ones(Y.shape)
for i in range(0, X.shape[0]):
    yhat[i] = thetas.dot(X[i,:])
    deltas[i] = yhat[i] - Y[i]
    delta_percent[i] = (deltas[i]/ Y[i]) * 100
total = 0
for i in range(0, X.shape[0]):
    total = total + np.abs(delta_percent[i])
total = total / X.shape[0]
print(total)
    
    
    
    
