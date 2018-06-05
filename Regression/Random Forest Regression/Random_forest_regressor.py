# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:42:52 2018

@author: Sai Teja Mummadi
"""

#Random Forest Regression

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
Rf = RandomForestRegressor(n_estimators=500,random_state= 0)
Rf.fit(X,y)

#Predicting the values
y_pred = Rf.predict(6.5)

#Visualizing
X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,Rf.predict(X_grid),color='blue')
plt.title('Using Random Forest')
plt.show()