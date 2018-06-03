# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 12:34:28 2018

@author: Sai Teja Mummadi
"""
#Simple linear Regression


#Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#Importing dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


#Splitting the data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)


#Fitting the linear regressor on test set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


#predicting values
y_pred=regressor.predict(X_test)


#visualizing training set 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()


#visualizing test set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()