# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 20:34:26 2018

@author: acer
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
"""
df = pd.read_csv("C:/Users/acer/.spyder-py3/BlackFriday.csv")
#filling in na values
df.fillna(0,inplace=True)
df = df.drop(['User_ID', 'Product_ID'], axis=1)
df = df.replace("F", 0)
df = df.replace("M", 1)
occ = pd.get_dummies(df['Occupation'])
city = pd.get_dummies(df['City_Category'])
pc1 = pd.get_dummies(df['Product_Category_1'])
pc2 = pd.get_dummies(df['Product_Category_2'])
pc3 = pd.get_dummies(df['Product_Category_3'])
df = df.drop(['Occupation', 'City_Category', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3'], axis=1)
df1 = pd.concat([df, occ, city, pc1, pc2, pc3], axis=1)

#Age
df1 = df1.replace("0-17", 9)
df1 = df1.replace("18-25", 22)
df1 = df1.replace("26-35", 32)
df1 = df1.replace("36-45", 42)
df1 = df1.replace("46-50", 48)
df1 = df1.replace("51-55", 53)
df1 = df1.replace("55+", 65)
scaler = MinMaxScaler()
scaler.fit(df1[['Age']])
MinMaxScaler(copy=True, feature_range=(0, 1))
df1[['Age']] = scaler.transform(df1[['Age']])

#stay in current city
df1 = df1.replace("4+", 10)
scaler2 = MinMaxScaler()
scaler2.fit(df1[['Stay_In_Current_City_Years']])
MinMaxScaler(copy=True, feature_range=(0, 1))
df1[['Stay_In_Current_City_Years']] = scaler2.transform(df1[['Stay_In_Current_City_Years']])

#Purchase min=185 max=23961
Purchase = np.array(df1['Purchase'].values.tolist())    
for i in range(0, len(Purchase)):
    if Purchase[i] < 8000:
        Purchase[i] = 0
    elif Purchase[i] < 16000:
        Purchase[i] = 1
    else:
        Purchase[i] = 2
Purchase = pd.DataFrame(Purchase,dtype=np.float)
df1 = df1.drop(['Purchase'], axis=1)

df1.to_csv('hw6_data_modified1.csv')
Purchase.to_csv('hw6_label1.csv')
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = pd.read_csv("C:/Users/acer/.spyder-py3/hw6_data_modified1.csv")
X = X.drop(['Unnamed: 0'], axis=1)
Y =pd.read_csv("C:/Users/acer/.spyder-py3/hw6_label1.csv")
Y = Y.drop(['Unnamed: 0'], axis=1)
x = np.array(X)
y = np.array(Y)
#亂數拆成訓練集(75%)與測試集(25%)
x_train, x_test, y_train, y_test = train_test_split(x, y, \
                                test_size=0.25, shuffle=True, random_state = 41)

#scaler = StandardScaler().fit(x_train)
#x_train_scaled = scaler.transform(x_train)
#x_test_scaled = scaler.transform(x_test)

clf = RandomForestClassifier(n_estimators=100, max_depth=30,
                             random_state=41)
modelrf = clf.fit(x_train, y_train.ravel())

y_predict = modelrf.predict(x_test)
y_predict.reshape(len(y_predict), 1)
y_predict1 = modelrf.predict(x_train)
y_predict1.reshape(len(y_predict1), 1)

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

precision, recall, accuracy = confusionmatrix(y_test, y_predict)
