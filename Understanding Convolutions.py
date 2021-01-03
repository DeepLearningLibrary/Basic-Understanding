# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:47:47 2021

@author: Grant
"""

import torch
import torch.nn as nn

#1D Convolutions
conv = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3) #2 channels, 16 kernels of size 3
print(conv.weight.size())
print(conv.bias.size())

x = torch.rand(1, 2, 64) #batch size 1, 2 channels, 64 samples
print(conv(x).size())

conv = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5) #2 channels, 16 kernels of size 5
print(conv(x).size())
print()

#2D Concolutions
conv = nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(3, 5)) #20 channels, 16 kernels, kernel size 3 x 5
x = torch.rand(1, 20, 64, 128) #1 sample, 20 channels, height 64, width 128
print(conv.weight.size())
print(conv(x).size())

conv = nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(3, 5), stride=1, padding=(1, 2)) #20 channels, 16 kernels of size 3 x 5, stride 1, padding of 1 and 2
print(conv(x).size())