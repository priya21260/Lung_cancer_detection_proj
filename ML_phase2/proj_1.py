# without sklearn

import numpy as np
import matplotlib.pyplot as plt

#for each it keeps x same 
np.random.seed(0)

# generating data
x = np.random.rand(100)
y = 3*x + 2 + np.random.randn(100)*0.1

# initializing parameter
w = 0
b = 0

lr = 0.01 # not too fast nor too slow
epochs = 1000

losses = []

#training loop
for i in range (epochs) :

    y_pred = w*x + b 

    # computing loss (MSE)
    loss = np.mean((y_pred - y)**2)
    losses.append(loss)

    #computing gradients

    dw = (2/len(x)) * np.sum((y_pred - y) * x )
    db = (2/len(x)) * np.sum(y_pred - y)

    # updating parameters
    w = w - lr * dw
    b = b - lr * db 

    if i%100==0 :
        print(f"Ephoch{i} , loss : {loss:.4f}")

# final parameters
print("\n Final parameters:")
print(f"W = {w:.4f},b = {b:.4f}")

#plot data + filled line
plt.scatter(x,y,label="Data")
plt.plot(x,w*x+b,color = "red",label = "Fitted Line")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()

# plot loss curve
plt.plot(losses)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

'''

# with sklearn 

# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 2: Create sample data (y = 3x + 2 + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1)

# Step 3: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create model
model = LinearRegression()

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Get learned parameters
m = model.coef_
b = model.intercept_

print("Slope (m):", m)
print("Intercept (b):", b)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Plot results
plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, label="Predicted", linewidth=2)
plt.legend()
plt.show()


'''