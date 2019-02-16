# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:53:28 2018

@author: User
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

"""

df = pd.read_csv("BlackFriday.csv")
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

資料前處理""" 

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
    accuracy = 	(cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[2][2]) / al
    return precision, recall, accuracy		
    		


X = pd.read_csv("hw6_data_modified1.csv")
X = X.drop(['Unnamed: 0'], axis=1)
Y =pd.read_csv("hw6_label1.csv")
Y = Y.drop(['Unnamed: 0'], axis=1)
x = np.array(X)
y = np.array(Y)

#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state = 41)

#調整參數
parameters = {'kernel':('linear', 'rbf'), 'C':[10, 100, 1000], 'gamma':[0.001, 0.0001]}

#cv = int, cross-validation generator or an iterable, optional
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameters, cv=5)

find_bestx = x_train[0:5000]
find_besty = y_train[0:5000]
find_besty = find_besty.ravel()
clf.fit(find_bestx, find_besty)

# Print out the results 
print('Best score for training data:', clf.best_score_)
print('Best `C`:',clf.best_estimator_.C)
print('Best kernel:',clf.best_estimator_.kernel)
print('Best `gamma`:',clf.best_estimator_.gamma)


# Create the SVC model 
svc_model = svm.SVC(gamma=0.001, C=100., kernel='rbf')
svc_model.fit(x_train, y_train.ravel())
svc_model.score(x_test, y_test)
p1_test = svc_model.predict(x_test)
#p2_train = svc_model.predict(x_train)

p_test, r_test, a_test = confusionmatrix(y_test, p1_test)
#p_train, r_train, a_train = confusionmatrix(y_train, p2_train)

