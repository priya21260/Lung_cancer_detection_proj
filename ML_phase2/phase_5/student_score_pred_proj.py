# project - 1 (basic data load) (linear regression)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np

data = pd.read_csv("hrs_score.csv")

x = data[["Hours"]]
y = data["Score"]

model = LinearRegression()
model.fit(x,y)

predicted_score = model.predict(x)

# evaluate
mae = mean_absolute_error(y,predicted_score)
mse = mean_squared_error(y,predicted_score)
rmse = np.sqrt(mse)

# show result 
print("Mean Absolute Error (MAE) :",mae)
print("Mean Squared Error (MSE) :",mse)
print("Root Mean Squared Error (RMSE) :",rmse)

new_hour = float(input("Enter a hour = "))
new_pred = model.predict(new_hour)
print(f"Prediction for {new_hour}hrs is score = {new_pred}")