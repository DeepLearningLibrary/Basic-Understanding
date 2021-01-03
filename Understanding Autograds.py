# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:12:30 2021

@author: Grant
"""

import torch

x = torch.tensor([[1, 2], [3, 4]], requires_grad=True, dtype=torch.float32)
y = x - 2
print(x.grad_fn)
print(y.grad_fn)
print(y.grad_fn.next_functions[0][0])
print(y.grad_fn.next_functions[0][0].variable) #x's gradient moved into y
print()

z = y * y * 3
a = z.mean()
print(z)
print(a)
print()

a.backward() #carry out backpropagation on a
print(x.grad)
print()

x = torch.randn(3, requires_grad=True)
y = x * 2
i = 0
while y.data.norm() < 1000:
    y = y * 2
    i += 1
print(y)
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients) #run backpropagation to get gradients
print(x.grad)
print(i) #2 ^ i gets the middle value of x.grad
print()

#both x and w allows gradient accumulation
n = 3
x = torch.arange(1., n + 1, requires_grad=True)
print(x)
w = torch.ones(n, requires_grad=True)
print(w)
z = w @ x
print(z)
z.backward() #runs backpropagation
print(x.grad, w.grad, sep='\n')
print()

#only w allows gradient accumulation
x = torch.arange(1., n + 1)
print(x)
w = torch.ones(n, requires_grad=True)
print(w)
z = w @ x
print(z)
z.backward() #runs backpropagation
print(x.grad, w.grad, sep='\n')

#all tensors without gradient accumulation
x = torch.arange(1., n + 1)
w = torch.ones(n)

with torch.no_grad():
    z = w @ x

#z.backward() #cannot do backpropgation without gradients


