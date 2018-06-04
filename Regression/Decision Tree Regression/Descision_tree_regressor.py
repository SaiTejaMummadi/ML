# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 21:28:43 2018

@author: Sai Teja Mummadi
"""

#Descision Tree Regression

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset= pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Descision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Predicting the values
regressor.predict(6.5)

#visualizing the Regressor
X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()