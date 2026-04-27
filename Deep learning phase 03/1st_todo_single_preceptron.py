# max() is python inbuilt np.maximum() is numpy built
# single neuron with no learning 
'''
import numpy as np

x = np.array([1,2,3]) # inputs
w = np.array([2,-1,3]) # weights
b = 1 # bias
z = np.dot(x,w) + b # neuron computaion
output = max(0,z) # activation(relu) gives (0 or greater no.)
print("output:",output)

'''
# single preceptron along with learning solving the AND gate
# single preceptron is only able to learn linear data
import numpy as np

# Activation function
def step(x) :
    return 1 if x>0 else 0

# training data
x = np.array([[0,0],[0,1],[1,0],[1,1]]) 
y= np.array([0,0,0,1]) # expected output

w = np.zeros(2) # weight
b = 0 # bias

lr = 0.1 # laerning rate

#training loop
for epoch in range (10):
    for i in range (len(x)):
        x_i = x[i]
        y_true = y[i]

        #Forward 
        z = np.dot(x_i,w) + b 
        y_pred = step(z)

        
        # Error
        error = y_true - y_pred

        # update rule
        w =  w + lr * error * x_i
        b = b + lr * error 

print("Finalweights :",w ," Finalbias :",b)

correct = 0

for i in range (len(x)):
        x_i = x[i]
        y_true = y[i]

        #Forward 
        z = np.dot(x_i,w) + b 
        y_pred = step(z)

        print("Input:",x_i ,"pred :",y_pred,"Actual:",y_true)

        if y_pred == y_true:
            correct += 1
    
accuracy = correct/len(x)

print("Acuuracy:",accuracy)
   
