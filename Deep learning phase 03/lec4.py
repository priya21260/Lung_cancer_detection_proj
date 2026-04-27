'''
Nueral Network Structure and Components
1.Dataloader 
   Data set loader
   MNIST classifier
2. nn.Module (neural network) 
   class structure 
   layers
   Activation functions
   optimiser
   saving Model
   nn.module -->1.box consisting of learning no.
                2.Rules --> instructions 
                3.Behaviour --> input based output 

    
Row tensors                                   trainable models
Numbers                                         numbers
line                                            organised data
here you meantion each and everything           automatically trace the learning no.
able to     break                                   clean structure 
                                                professional 




# Class Structures
import torch 
import torch.nn as nn

class myModel(nn.Module):
    def __init__(self):  
       super().__init__()
    
self.w = nn.parameter(torch.tensor(0.0))

defforward(self,x)
returnself.w*x



# layers
weight     
input importance                    bias
big --> max                      up/down
small --. less                   a small adjustment


shape rules --> 1.input -> match with input feature
            --> 2.output size fixed
            --> 3.batch size --> hamesha grp me ayega

import torch 
import torch.nn as nn
layer = nn.linear(3,2) # 3 --> no. of input , 2--> no. of output
x = torch.tensor ([1.0,2.0,3.0])
y = layer(x)
print(y)  

output -->
tesor([[0.0465,-1.6146]],grad_fn<AddmmBackwaed0>)
'output = ("input * weights) + bias \nnn.Linear (32,3)\n[?,?,?] 3 numbers\n [?,?]



# Activation Functions
after a layer 
* nn.Relu -> rectifier linear unit  -ve --> 0 , +ve --> keep 
* nn.sigmoid --> yes/no  convert blw (0 and 1) --> use in final output only
*nn.softmax --> [2,1,0] --> [0.7,0.2,0.1] --. digit recognition 

Hidden layer --> nn.Relu
Binary output -- sigmoid
MultiClass output --> softmax

linear --> relu --> linear --> softmax / sigmoid --> output

# Optimiser
1. SGD 
syntax --> torch.optim.SGD(model.parameters(),lr=0.01)
2.Adam
torch.optim.Adam(model.parameters(),lr=0.001) # handel the noisy data better

optimiser.zero_grad()
forward pass
loss calcn
loss.backward()
optimiser.stop()

'''

import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
      super.__init__()
      self.Linear=nn.Linear(1,1)
    def forward(self,x):
     return self.linear(x)
    
x = torch.tensor([[1.],[2.],[3.],[4.]])
y = torch.tensor([[2.],[4.],[6.][8.]])
model = MyModel()
loss_fn = nn.MSELoss()
optimiser = optim.Adam(model.parameters(),lr = 0.1)
for epoch in range (10):
    optimiser.zero_grad()
    y_pred = model()
    loss = loss_fn(y_pred,y)
    loss.backward()
    optimiser.step()
    print(f"Epoch (epoch+1),Loss: {loss.item():.4f}")

torch.save(model.state_dict(),"Linear_model.pth")

'''
1. DataSet
    50,000 images  --> problem  1. memory explode
                                2. unstable learning
                                3. slow and inefficent  

    use Batch --> 1. fit in memory
                  2. Better learning
                  3. faster training

  dataset -> 1sample
  data loader > grp into batches
  model -> learn batch wise

'''

