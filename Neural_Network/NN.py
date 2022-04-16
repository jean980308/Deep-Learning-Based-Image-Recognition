# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:35:02 2022

@author: jedi
"""

import numpy ,random ,os

lr = 1
bias= 1
weights = [random.random() , random.random() ,random.random() ,random.random() ]
weights

def Perceptron(input1 , input2 ,input3 , output):
  outputP= input1*weights[0] + input2*weights[1] +input3*weights[2] +bias*weights[3]

  outputP= 1/(1+numpy.exp(-outputP))

  error = output -outputP

  weights[0] += error * input1 * lr
  weights[1] += error * input2 * lr
  weights[2] += error * input3 * lr
  weights[3] += error * bias  * lr
  return weights , error ,outputP



weights_log= numpy.zeros((50,8,4),dtype='float')
error_log = numpy.zeros((50,8,1),dtype='float')
outputP_log= numpy.zeros((50,8,1),dtype='float')

for i in range(50):

  weights_log[i][0], error_log[i][0], outputP_log[i][0] = Perceptron(1,1,1,1)
  weights_log[i][1], error_log[i][1], outputP_log[i][1] = Perceptron(1,1,0,1)
  weights_log[i][2], error_log[i][2], outputP_log[i][2] = Perceptron(1,0,1,1)
  weights_log[i][3], error_log[i][3], outputP_log[i][3] = Perceptron(1,0,0,1)
  weights_log[i][4], error_log[i][4], outputP_log[i][4] = Perceptron(0,1,1,1)
  weights_log[i][5], error_log[i][5], outputP_log[i][5] = Perceptron(0,1,0,1)
  weights_log[i][6], error_log[i][6], outputP_log[i][6] = Perceptron(0,0,1,1)
  weights_log[i][7], error_log[i][7], outputP_log[i][7] = Perceptron(0,0,0,0)
  
  


print(weights_log)



logs = numpy.zeros((50) ,dtype='float')
for i in range(50):
  logs[i] = error_log[i].mean()
  
print(logs)

print(weights_log)

print(error_log)

print(outputP_log)

print(Perceptron)
