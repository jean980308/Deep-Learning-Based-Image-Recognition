# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:28:34 2022

@author: jedi
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

fig_1 = plt.figure()

kmeans_plot=KMeans(n_clusters=6 , init='k-means++' , max_iter=300 , n_init=10 , random_state=0)
X , y =make_blobs(n_samples=300 ,centers=6 ,cluster_std=0.60 , random_state=0) 
pred_y=kmeans_plot.fit_predict(X)
plt.scatter(X[:,0],X[:,1]) 
plt.scatter(kmeans_plot.cluster_centers_[:,0] ,kmeans_plot.cluster_centers_[:,1],s=300,c='red')

plt.show()




fig_2 = plt.figure()

wcss = []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i , init='k-means++' , max_iter=300 , n_init=10 , random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

