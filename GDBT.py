#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 22:06:29 2019

@author: wuhuanxuan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import confusion_matrix

def confusionmatrix(true, predict):
    cnf_matrix = confusion_matrix(true, predict)
    al = 0
    for i in range(3):
       for j in range(3):
            al = al + cnf_matrix[i][j]
    P1 = cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0]+cnf_matrix[2][0])
    P2 = cnf_matrix[1][1]/(cnf_matrix[0][1]+cnf_matrix[1][1]+cnf_matrix[2][1])
    P3 = cnf_matrix[2][2]/(cnf_matrix[0][2]+cnf_matrix[1][2]+cnf_matrix[2][2])
    R1 = cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[0][1]+cnf_matrix[0][2])
    R2 = cnf_matrix[1][1]/(cnf_matrix[1][0]+cnf_matrix[1][1]+cnf_matrix[1][2])
    R3 = cnf_matrix[2][2]/(cnf_matrix[2][0]+cnf_matrix[2][1]+cnf_matrix[2][2])
    precision = (P1+P2+P3)/3
    recall = (R1+R2+R3)/3
    accuracy = (cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[2][2])/ al
    return precision, recall, accuracy

#data import
X = pd.read_csv("hw6_data_modified1.csv")
X = X.drop(['Unnamed: 0'], axis = 1)
Y = pd.read_csv("hw6_label1.csv")
Y = Y.drop(['Unnamed: 0'], axis = 1)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, shuffle=True)
clf = ensemble.GradientBoostingClassifier(max_depth = 6)
clf.fit(train_X,train_Y)
predict = clf.predict(test_X)
accuracy = clf.score(test_X,test_Y)

precision, recall, accuracy = confusionmatrix(test_Y, predict)
print(precision)
print(recall)
print(accuracy)
