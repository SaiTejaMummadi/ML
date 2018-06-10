# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 14:23:08 2018

@author: Sai Teja Mummadi
"""

# k-means clustering

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:,[3,4]].values

#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',random_state=39)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('wcss')
plt.show()

# Fitting the dataset to Kmeans
kmeans = KMeans(n_clusters=5, init='k-means++',random_state=39)
y_kmeans=kmeans.fit_predict(X)

#Visualizing

plt.scatter(X[y_kmeans==0 , 0], X[y_kmeans == 0, 1] , s=100, c='red', label='cluster1')
plt.scatter(X[y_kmeans==1 , 0], X[y_kmeans == 1, 1] , s=100, c='blue', label='cluster2')
plt.scatter(X[y_kmeans==2 , 0], X[y_kmeans == 2, 1] , s=100, c='cyan', label='cluster3')
plt.scatter(X[y_kmeans==3 , 0], X[y_kmeans == 3, 1] , s=100, c='pink', label='cluster4')
plt.scatter(X[y_kmeans==4 , 0], X[y_kmeans == 4, 1] , s=100, c='green', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],s=300, c='magenta',label='centroids')
plt.xlabel('income')
plt.ylabel('Spending')
plt.title('Clusters of customers')
plt.legend()
plt.show()

