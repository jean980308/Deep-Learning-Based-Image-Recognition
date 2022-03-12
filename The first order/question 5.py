# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:34:41 2022

@author: jedi
"""
import time
import torch

device_cpu= torch.device("cpu")
device_gpu= torch.device("cuda")

time_start_cpu = time.time()    #結束計時

e_cpu= torch.randint(-128,128,(500,500) , device=device_cpu)

print(e_cpu)

f_cpu= torch.randint(-128,128,(500,500) , device=device_cpu)

print(f_cpu)







result_cpu=abs(e_cpu)*f_cpu-e_cpu

time_end_cpu = time.time() 

time_cpu= time_end_cpu - time_start_cpu 

print(time_cpu)




time_start_gpu = time.time()    #結束計時


e_gpu= torch.randint(-128,128,(500,500) , device=device_gpu)

print(e_gpu)

f_gpu= torch.randint(-128,128,(500,500) , device=device_gpu)

print(f_gpu)  

result_gpu=abs(e_gpu)*f_gpu-e_gpu


time_end_gpu = time.time() 

time_gpu= time_end_gpu - time_start_gpu 

print(time_gpu)
