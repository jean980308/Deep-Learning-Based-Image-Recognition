# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch

available_gpus=[torch.cuda.device(i) for i in range (torch.cuda.device_count())]

print(available_gpus)

num_of_gpus=[torch.cuda.device(i) for i in range (torch.cuda.device_count())]

print(num_of_gpus)

for i in range(torch.cuda.device_count()):
  print(torch.cuda.get_device_name(i))