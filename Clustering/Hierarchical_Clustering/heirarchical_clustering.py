# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:56:00 2018

@author: Sai Teja Mummadi
"""
#Hierarchical Clustering

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Using Dendrogram to find the optimal number of customers
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian Distances')
plt.show()

#Fitting Heirarchical Cluster to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

#Visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc == 0, 1] , s=100, c='red', label='cluster1')
plt.scatter(X[y_hc==1,0],X[y_hc == 1, 1] , s=100, c='blue', label='cluster1')
plt.scatter(X[y_hc==2,0],X[y_hc == 2, 1] , s=100, c='green', label='cluster1')
plt.scatter(X[y_hc==3,0],X[y_hc == 3, 1] , s=100, c='cyan', label='cluster1')
plt.scatter(X[y_hc==4,0],X[y_hc == 4, 1] , s=100, c='yellow', label='cluster1')
plt.title('Clusters if customer')
plt.xlabel('Income')
plt.ylabel('spend score')
plt.legend()
plt.show()