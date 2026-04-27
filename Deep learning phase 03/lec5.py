'''

CNN --> convolutional nueral network

* edges,pattern,curves,shape,texture

filter --> smaller part of the image

  1st layer --> detects lines/ edges
 
  2nd layer --> combines the curve and corners
 
  3rd layer --> combine shapes into digits/ objects

nn.Convo2d(in_channels,out_channels,kernel_size,stride,padding)
 
 in_channels --> how many layers came in 
     black & white --> 1
     color RBG --> 3

   shape --> [batch,channels,height,width] --> [1,1,28,28]

  out_channels --> kinte pattern ap detect karna chahate ho
      each output channel == 1 pattern detector
      8 
      [1,1,28,28] --> [1,8,?,?] -->

  kernel_size
       decide the filter size 
        3 = 3*3 pixels
        5 = 5*5 pixels
        bigger the kernal --> it will see more context
        smaller the size of the kernal --> fine details detect             

  stride --> how you want it to move --> 1,2,3 means jumps
  if jump is larger --> outcome will be smaller , sone data will get lost

  padding --> should edge or not --> decide 
  if edging not done --> 1. filter cannot fully sit on edge
                         2. image will shrink
   to do edging --> write --> padding = 1


ex    [1,1,28,28]

conv = nn.conv2d(
   in_channel = 1,
   out_channel = 8,
   kernal = 3,
   stride = 1,
   padding = 1,
)  
 output --> [1,8,28,28] --> padding = 0 -->[1,8,26,26]
 stride  2 --> [1,8,14,14] 
 
'''

import torch 
import torch.nn as nn

image = torch.randn(1,1,28,28)

conv = nn.Conv2d(
    in_channels = 1,
    out_channels= 8,
    kernel_size = 3,
    stride = 1,
    padding = 1
)

output = conv(image)
print("Input image :",image.shape) # [1,1,28,28]
print("output image:"output.shape) # [1,8,28,28]


''' 
pooling --> keeps the most important/strongest/dominating signal and discard the other

ex = 1,2,3,5 --> it will keep 5 

pooling -> just selects not learns

'''
import torch 
import torch.nn as nn
x = torch.randn(1,8,28,28)
pool = nn.maxPool2d(kernal_size=2)
y = pool(x)
print("Before pooling",x.shape) # ([1,8,28,28])
print("After pooling",y.shape) # ([1,8,14,14])


'''
image --> conv2d --> Relu --> Pool --> conv2d --> Relu --> Pool --> Flatten(pattern to list)
--> linear --> output

'''
