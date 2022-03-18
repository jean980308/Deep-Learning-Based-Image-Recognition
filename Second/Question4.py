# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:31:54 2022

@author: jedi
"""

from __future__ import division , print_function
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


colors= ['b' ,'orange' ,'g' ,'r' ,'c' ,'m' ,'y' ,'k' ,'Brown' , 'ForestGreen']

centers=[[4.2],
         [1,7],
         [5,6]]
         

sigmas=[[0.8,0.3],
        [0.3,0.5],
        [1.1,0.7]]

np.random.seed(42)
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
#print('check',zip(centers,sigmas))
#print('check')
#for i in enumerate(zip(centers,sigmas)):
for i, ((xmu,ymu), (xsigma,ysigma)) in enumerate(zip(centers,sigmas)):
    
    xpts = np.hstack((xpts, np.random.standard_normal(200)*xsigma+xmu))
    print()
    ypts = np.hstack((ypts, np.random.standard_normal(200)*ysigma+ymu))
    labels= np.hstack((labels, np.ones(200)*i))



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
        
        