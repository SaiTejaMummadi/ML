# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:26:22 2018

@author: Sai Teja Mummadi
"""
#Polynominal Linear Regression

#Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#BUilding a linear regressor
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Building a Polynomial Regressor
from sklearn.preprocessing import PolynomialFeatures
Pf = PolynomialFeatures(degree=3)
X_poly=Pf.fit_transform(X)

poly_reg=LinearRegression()
poly_reg.fit(X_poly,y)

#Visualizing Linear Regression
plt.scatter(X,y,color='red')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.title('salary vs Xp')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.show()


#Visualizing Polynomial Regression
plt.scatter(X,y,color='red')
plt.plot(X,poly_reg.predict(X_poly),color='blue')
plt.xlabel("years of xp")
plt.ylabel('salary')
plt.title('salary vs xp')
plt.show()

#Predicting the data using Linear Regression
lin_reg.predict(7)

#Predicting the data using Polynomial Regression
"""Here we are using this fit trasform function cos we get a dimension error"""
poly_reg.predict(Pf.fit_transform(7))

