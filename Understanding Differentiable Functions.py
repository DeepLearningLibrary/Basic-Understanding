# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:55:37 2021

@author: Grant
"""

import torch
import numpy

class MyAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        ctx.save_for_backward(x1, x2)
        return x1 + x2
    
    @staticmethod
    def backward(ctx, grad_output):
        x1, x2 = ctx.saved_tensors
        grad_x1 = grad_output * torch.ones_like(x1)
        grad_x2 = grad_output * torch.ones_like(x2)
        return grad_x1, grad_x2

class MySplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x1 = x.clone()
        x2 = x.clone()
        return x1, x2
    
    @staticmethod
    def backward(ctx, grad_x1, grad_x2):
        x = ctx.saved_tensors[0]
        return grad_x1 + grad_x2

class MyMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        maximum = x.detach().numpy().max()
        argmax = x.detach().eq(maximum).float()
        ctx.save_for_backward(argmax)
        return torch.tensor(maximum)
    
    @staticmethod
    def backward(ctx, grad_output):
        argmax = ctx.saved_tensors[0]
        return grad_output * argmax


"""
x1 = torch.randn((5), requires_grad=True)
x2 = torch.randn((5), requires_grad=True)
print(f'x1: {x1}')
print(f'x2: {x2}')
myadd = MyAdd.apply  # aliasing the apply method
y = myadd(x1, x2)
print(f' y: {y}')
z = y.mean()
print(f' z: {z}, z.grad_fn: {z.grad_fn}')
z.backward() #carries out backpropagation
print(f'x1.grad: {x1.grad}')
print(f'x2.grad: {x2.grad}')
"""

"""
x = torch.randn((5), requires_grad=True)
print(f'x: {x}')
mysplit = MySplit.apply  # aliasing the apply method
x1, x2 = mysplit(x)
print(f' x1: {x1}')
print(f' x2: {x2}')
y = x1 + x2
print(f' y: {y}')
z = y.mean()
print(f' z: {z}, z.grad_fn: {z.grad_fn}')
z.backward() #carries out backpropagation
print(f'x.grad: {x.grad}')
"""

x = torch.randn((5), requires_grad=True)
print(f'x: {x}')
mymax = MyMax.apply  # aliasing the apply method
y = mymax(x)
print(f' y: {y}, y.grad_fn: {y.grad_fn}')
y.backward()
print(f' x.grad: {x.grad}')