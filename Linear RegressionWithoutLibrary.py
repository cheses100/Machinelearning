#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 15:04:55 2018

@author: danielmizrahi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Salary_Data.csv')

M = dataset.as_matrix()

X = np.ones(M.shape)
X[:,1] = M[:,0]
Y = M[:,1]

thetas = []



thetas = np.array([1,1], dtype=float)
print(thetas)
alpha = .001
yhat = np.ones(Y.shape)
sse = np.zeros(10000)
for i in range(0,10000):

    for j in range(0, X.shape[0]):
        for k in range(0, X.shape[1]):
            thetas[k] = thetas[k] - alpha*(thetas.dot(X[j,:]) - Y[j])*X[j,:][k]
        yhatt = thetas.dot(X[j,:])
        sse[i] = sse[i] + (yhatt - Y[j])**2
for i in range(0, X.shape[0]):
    yhat[i] = thetas.dot(X[i,:])
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], yhat)
plt.show()

t = range(0, 10)
plt.plot(t, sse[0:10])
plt.show()
    
    
    
