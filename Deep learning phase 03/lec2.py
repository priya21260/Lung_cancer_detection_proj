# Tensor's setup

'''
NN --> work with float not int
Tensor --> storing the number in a box with smart way
Tensor --> number + learning ability + can go to GPU
GPU -->10000 same worker --> fast
CPU --> fast but limited

pytorch liberary --> tensor
# 1. pip3 install torch
'''

import torch 
x = torch.tensor([1,2,3,4,5]) # torch--> lib , tensor --> func
print(x) # 1D tensor

y = torch.tensor([
    [1,2],
    [3,4]
])
print(y) # 2D tensor

torch.zeros(3,2) # 3 rows , 2 columns --> initilised by 0

o = torch.ones(2,2)
print(o) # 2-> row 2-> column initialised by 1

r =torch.rand(2,2)
print(r) # 2->row 2->column with random numbers

a = torch.tensor([
    [1,2],
    [2,3],
    [4,5]
])
print(a.shape) # (3,2), 3-->row , 2--> column

# indexing
i= torch.tensor([10,20,30]) # 1D tensor
print(i[1]) # 20
print(i[0]) # 10

i_2 = torch.tensor([
    [3,56,50],
    [34,56,45],
])
print(i_2[0,1]) # 56 [row,column]

a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
print(a+b) # ([5,7,9])
print(a*b) # ([4,10,18]) also for - , / , %
print(a*2) # ([2,4,6]) --> all multipied by 2

# tensor and numpy --> share memory 

import numpy as np
arr = np.array([1,2,3])
n = torch.from_numpy (arr) # tensor lso points towaerd same memory

# LINEAR ALGEBRA BASED MINI PROJECT
import torch
x = torch.tensor([1.,2.,3.,4.]) # input 
y = torch.tensor([2.,4.,6.,8.]) # actual output
w = torch.tensor(0.0) # weight
b = torch.tensor(0.0) # bias

y_pred = w*x + b

loss = ((y_pred - y)**2).mean() # MSE 

print("predicted values",y_pred)
print("Actual values",y)
print("loss",loss)










