# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:10:33 2022

@author: jedi
"""

import numpy as np
import scipy.signal

def convolved2d_2(image, kernel):
    
    kernel = np.flipud(np.fliplr(kernel))  
     
    """  flipud = flip upside down  fliplr=Flip left and right """
    
    output = np.zeros_like(image)
     
    """ np.zeros_like similar to matlab zero """
    
    
    image_padded=np.zeros((image.shape[0]+2 ,image.shape[1] +2))
    
    k = kernel.shape[0]

    
    """img.shaope[0] = high ,img.shape[1]=width """
    
    image_padded[1:-1, 1:-1] =image
    
    for x in range(0,image.shape[1]):
        for y in range(0,image.shape[0]):
            
            mat = image_padded[x:x+k, y:y+k]
            output[x,y] =np.sum(np.multiply(mat,kernel))
            """
            Here I  have encounter error setting an array element with a sequence 
            , because Left side and right side data dimension is different 
            , That will cause to erroring
            """

    return output
    
image=np.random.randint(0,255,size=(6,6))
kernel= np.array ([[1, 0, 0 ],[0, 1, 0] , [0, 0, 1]] ) 

my_writing=convolved2d_2(image,kernel)

print("my algorithm")
print(my_writing )
"""
This is my algorithm generated answer
"""
right_answer=scipy.signal.convolve2d(image,kernel,mode='same', boundary='fill', fillvalue=0)

print("ANSWER")
print(right_answer)
"""
This is use python genrated answer
"""
