import numpy as np

# activation relu
def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Data for XOR problem
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
    
np.random.seed(0)
w1 = np.random.randn(2,4) # 2 row 4 column 
b1 = np.zeros((1,4)) # [1,1,1,1]
w2 = np.random.randn(4,1) 
b2 = np.zeros((1,1)) #[1]

lr = 0.1

for epochs in range (10000):
    z1 = np.dot(x,w1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1,w2) + b2
    
    y_pred = sigmoid(z2)

    # MSE (mean square error)
    loss = np.mean((y-y_pred)**2)

    d_loss = (y_pred - y)                          # (4,1)
    d_z2 = d_loss * sigmoid_derivative(z2)         # (4,1)

    # Gradients for w2, b2
    dw2 = np.dot(a1.T, d_z2)                       # (4,4)^T * (4,1) = (4,1)
    db2 = np.sum(d_z2, axis=0, keepdims=True)

    # Backprop into hidden layer
    d_a1 = np.dot(d_z2, w2.T)                      # (4,1)*(1,4) = (4,4)
    d_z1 = d_a1 * relu_derivative(z1)              # (4,4)

    # Gradients for w1, b1
    dw1 = np.dot(x.T, d_z1)                        # (2,4)
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    w2 -= lr * dw2
    b2 -= lr * db2

    w1 -= lr * dw1
    b1 -= lr * db1

    if epochs % 1000 == 0:
       print(f"Epoch {epochs}, Loss: {loss:.4f}")

print("\nFinal Predictions:")
y_pred = sigmoid(np.dot(relu(np.dot(x, w1) + b1), w2) + b2)

for i in range(len(x)):
    print(f"Input: {x[i]} -> Pred: {y_pred[i][0]:.4f} -> Rounded: {round(y_pred[i][0])}")