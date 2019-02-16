# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:20:10 2019

@author: Yvonne
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

def confusionmatrix(true, predict):
    cnf_matrix = confusion_matrix(true, predict)
    al = 0
    for i in range(3):
        for j in range(3):
            al = al + cnf_matrix[i][j]
    precision0 = cnf_matrix[0][0] / (cnf_matrix[0][0] + cnf_matrix[1][0] + cnf_matrix[2][0])
    precision1 = cnf_matrix[1][1] / (cnf_matrix[0][1] + cnf_matrix[1][1] + cnf_matrix[2][1])
    precision2 = cnf_matrix[2][2] / (cnf_matrix[0][2] + cnf_matrix[1][2] + cnf_matrix[2][2])
    recall0 = cnf_matrix[0][0] / (cnf_matrix[0][0] + cnf_matrix[0][1] + cnf_matrix[0][2])
    recall1 = cnf_matrix[1][1] / (cnf_matrix[1][0] + cnf_matrix[1][1] + cnf_matrix[1][2])
    recall2 = cnf_matrix[2][2] / (cnf_matrix[2][0] + cnf_matrix[2][1] + cnf_matrix[2][2])
    precision = (precision0 + precision1 + precision2) / 3
    recall = (recall0 + recall1 + recall2) / 3
    accuracy = (cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[2][2]) / al
    return precision, recall, accuracy, cnf_matrix


X = pd.read_csv('D:/桌面備/大四/dm/期末project/hw6_data_modified1.csv')  #將資料讀取進來
X = X.drop(['Unnamed: 0'], axis=1)
Y = pd.read_csv('D:/桌面備/大四/dm/期末project/hw6_label1.csv')  #將資料讀取進來
Y = Y.drop(['Unnamed: 0'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, shuffle=True, random_state = 41)
classifier = DecisionTreeClassifier(max_depth = 9)  
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
precision, recall, accuracy,confusion = confusionmatrix(y_test, y_pred)
print("accuracy：",accuracy)
print("precision：",precision)
print("recall：",recall)
