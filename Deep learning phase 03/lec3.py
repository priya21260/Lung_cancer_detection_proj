# auto grad --> loss,responsible ,fixees them automatically

'''
s1 --> guess
s2 --> measure the loss
s3 --> automatically fix
s4 --> apply the changes --> s1 (loop)

1. required grad

rool no    name     maths marks 
fixed      fixed     can change (impact)

# so we have to provide explicitly what are provided for learning
[syntax --> requires_grad = True ]
   if false then it ignores it
   else it can automate on it 

'''

import torch 
# requires_grade = True / False
w = torch.tensor(2.0,requires_grad=True)
 # tracks tha value(2.0) and if needed to automate then it can do it
x= torch.tensor(4.0)
y = w*x
print(y)  # tensor(12., grad_fn=<MulBackward0>) --> true 
# remebers hoe this no. is formed long with you can later ask where the mistake took place
# false --> tensor(12.)

'''
2. Computation graph (automatic)

flour --> sugar --> milk --> saked it --> cake is formed 
if sugar --> jyada hua to mistake find karne ke liye steps malum rehena jaruri he
fix karsakteho milk increase karke

'''
x = torch.tensor(2.0)
w = torch.tensor(3.0,requires_grad = True)
y = w*x # computaional graph--> x->fix , w-. kina loss kiya usko store kareke rakega
print(y)

'''
3. Backward () --> how much each learnable no. contriuted to mistake
  1. who is responsible for the mistake 
  2. How much resposible
ch1 ch2 ch3 ch4 ch5   60/100
yeha mistake kis chap me ur kitna hua he ye teaches aur student janege

a. loss no.
b. requires_grad
c. computation graph 

all this 3 are needed for BackWard ()

* works with only single grap --> mean can't use it 2 or more time in same code
--> you can use w.grad.zero_()
'''
import torch
x = torch.tensor(2.0)
w = torch.tensor(3.0,requires_grad=True)
y_pred = w*x
y_true = torch.tensor(10.0)
loss = (y_pred-y)**2
print(loss) # tensor(16.,grad_fn=<PowBackward0>)
loss.backward()
print(w.grad) # tensor(-16.)--> -w --> increase the w , w --> decrease the w
# (.grad --> can be only used for learning value


'''
4. .grad (kis direction me move karna aur ktna --> stores)
(just gives advice --. doesn't change anything)

salty he (1.)--> salt ko remove karu 
             --> salt ko dalu 
         (2.) kitna (remove ya add karu)

w.grad > 0 --> decrease w
w.grad < 0 --> increase the w

'''

'''
5. .gradient
gradient descent --> improve the no. using the .grad advice

# new value = old_value - (learning rate * gradient)
   old_value = current output
   gradient = .grad
   learning rate = 0.1(normal)

'''

import torch 
x = torch.tensor(2.0)
w = torch.tensor(3.0,requires_grad = True)
y_true = torch.tensor(10.0)
lr = 0.1 # 10%
y_pred = w*x
loss = (y_pred - y)**2
loss.backward()
print('Before update')
print("w:",w.item()) # bec they are object so item is used
print("loss",loss.item())
print('gradient',w.grad.item())

with torch.no_grad(): # do not track
    w -= lr*w.grad  # w = w - (lr*w.grad)
# important reset the gradient
w.grad.zero_()
print("After update")
print("w",w.item())


#  what if we won't use reset
x = torch.tensor(2.0)
w = torch.tensor(3.0,requires_grad=True)
y = torch.tensor(10.0)
loss1 = (w*x - y)**2
loss1.backward()
print("After 1st backward",w.grad.item()) # -16.0 --> output 
# so after it add the w.grad.zero_() 
loss2 = (w*x - y)**2
loss2.backward() # --> no updated so .grad doesn't change
print("After 2nd backward",w.grad.item()) # -32.0 --> output

'''
6. Training loop 
  
  prediction --> measure loss --> backward(find mistake) --> 
  fixkarenge mistake ko --> clear karenge paile mistake ko / apply

'''
import torch 
x = torch.tensor([1.,2.,3.,4.])
y_true = torch.tensor([2.,4.,6.,8.])
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1
epochs = 10

for epoch in range (epochs):
    # step1 prediction 
    y_pred = w*x
    # step2 loss calulate
    loss = (y_pred - y**2).mean()
    # step3 mistake find 
    loss.bakward() 
    # step4 uodate
    with torch.no_grad():
        w -= lr*w.grad
    print(f"Epoch {epoch + 1}: w={w.item():.4f},loss = {loss.item}:.4f")

'''
                normal         no_grad
meamory use->     yes              no
graph built ->    yes              no
grad calculate    yes              no
used for          training         updating

 .4f --> after decimal 4 digit
'''



