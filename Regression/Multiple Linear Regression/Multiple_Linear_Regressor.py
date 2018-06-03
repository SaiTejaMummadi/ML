# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 21:38:08 2018

@author: Sai Teja Mummadi
"""
#Multiple Linear Regression

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#Ecoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lbl_encod_X = LabelEncoder()
X[:,3] = lbl_encod_X.fit_transform(X[:,3])
ohe = OneHotEncoder(categorical_features=[3])
X=ohe.fit_transform(X).toarray()  

#Taking care of dummy variable
X=X[:,1:]

#Splitting the data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Multiple Linear Regressor

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(X_train,y_train)

#predicting using test set

y_pred= mlr.predict(X_test)

#optimising  the Multiple linear regression model

import statsmodels.formula.api as sm
X= np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt= X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt= X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt= X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt= X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

