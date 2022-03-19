# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 10:16:01 2022

@author: jedi
"""
import numpy as np
from matplotlib import pyplot as plt
import torch


"""
I change value 1000 to 400 times , Because my computer is not very good

"""


def mse(true,pred):
    """
    

    Parameters
    ----------
    true : TYPE
        DESCRIPTION.
    pred : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    return np.sum(np.abs(true-pred)**2)

fig, ax1 = plt.subplots(1,1, figsize = (7,5))

# array of same target value 400 

target = np.repeat(100 , 400)
# np.arange is equal -400:2:400

pred = np.arange(-400,400, 2)

loss_mse = [mse(target[i], pred[i]) for i in range(len(pred)) ]



#plot
ax1.plot(pred, loss_mse)
ax1.set_xlabel('Predictions')
ax1.set_ylabel('Loss')
ax1.set_title("MSE Loss numpy vs. Predictions")


fig.tight_layout()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Tensor Version
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def mse_tensor(true, pred):
    
    
    
    return torch.sum((true - pred)**2)





# change np number array to tensor number array
input= torch.autograd.Variable(torch.from_numpy(target))
target= torch.autograd.Variable(torch.from_numpy(pred))

# plot tesnor version MSE
fig2, ax2 = plt.subplots(1,1, figsize = (7,5))


"""
Use Tensor function==>torch.nn.L1Los to implment
"""
loss_fn = torch.nn.MSELoss(reduction='none')
loss = [loss_fn(input[i].float(), target[i].float()) for i in range(len(pred))]



#plot
ax2.plot(pred, loss)
ax2.set_xlabel('Predictions')
ax2.set_ylabel('Loss')
ax2.set_title("MSE tensor use function Loss vs. Predictions")



fig2.tight_layout()

"""
Use we create funnction 

"""

fig3, ax3 = plt.subplots(1,1, figsize = (7,5))

loss_ten = [mse_tensor(input[i], target[i]) for i in range(len(pred)) ]



ax3.plot(pred, loss_ten)
ax3.set_xlabel('Predictions')
ax3.set_ylabel('Loss')
ax3.set_title("MSE tensor I Define Loss vs. Predictions")


fig3.tight_layout()







































