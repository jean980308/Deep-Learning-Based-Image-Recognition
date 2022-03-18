# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:56:25 2022

@author: jedi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

col_names= ['sepal_length' , 'sepal_width' , 'petal_length' , 'petal_width' , 'categories']
iris = pd.read_csv(url,header=None , names=col_names)

###plot ###

category_1=iris[iris['categories']=='Iris-setosa']
category_2=iris[iris['categories']=='Iris-virginica']
category_3=iris[iris['categories']=='Iris-versicolor']


fig ,ax =plt.subplots()
ax.plot(category_1['petal_length'] ,category_1['petal_width'] ,marker='o' , linestyle='' ,ms=12 ,label='Iris-setosa')
ax.plot(category_2['petal_length'] ,category_2['petal_width'] ,marker='o' , linestyle='' ,ms=12 ,label='Iris-virginica')
ax.plot(category_3['petal_length'] ,category_3['petal_width'] ,marker='o' , linestyle='' ,ms=12 ,label='Iris-versicolor')



ax.legend()
plt.show()


##################

iris_class = {'Iris-setosa' :0, 'Iris-virginica' :1 , 'Iris-versicolor' :2 }
iris['labels'] = [iris_class[i] for i in iris.categories]

X= iris.drop(['categories','labels'] ,axis=1)
Y= iris.labels

from sklearn.model_selection  import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , Y ,random_state=1)


from sklearn.neighbors import KNeighborsClassifier

knn =  KNeighborsClassifier (n_neighbors=10)

knn.fit(X_train, Y_train)


print(knn.score(X_test, Y_test))
''''
fig1 ,axes1 =plt.subplots(3,3,figsize=(8,8))
alldata = np.vstack((xpts , ypts))
fpcs = []


for ncenters ,ax in enumerate(axes1.reshape(-1) ,2) :
    
    cntr , u ,u0 , d , jm , p ,fpc =fuzz.cluster.cmeans ( alldata, ncenters, 2 ,error=0.005 ,maxiter=1000 , init=None)
    
    fpcs.append(fpc)
    
    cluster_membership=np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j] ,
                ypts[cluster_membership == j] , '.' ,color=colors[j] ) 
    
    for pt in cntr:
        ax.plot(pt[0] ,pt[1], 'rs')
        
    ax.set_title('Centers ={0}; FPC={1:.2f})'.format(ncenters , fpc))
    ax.axis('off')
    
fig1.tight_layout()

'''
for i ,((xmu,ymu),(xsigma , ysigma)) in enumerate(zip(centers,sigmas)):
    
    xpts = np.hstack((xpts, np.random.standard_normal(200)*xsigma+xmu))
    print()
    ypts = np.hstack((ypts, np.random.standard_normal(200)*ysigma+ymu))
    labels= np.hstack((labels, np.ones(200)*i))

''''