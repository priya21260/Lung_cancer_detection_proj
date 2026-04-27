# project - 1 (basic data load) (linear regression)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("Students performance Dataset.csv")

x = data[["Study_Hours_per_week"]]
y = data["Final_Score"]

model = LinearRegression()
model.fit(x,y)

predicted_score = model.predict(x)

# evaluate
mae = mean_absolute_error(y,predicted_score)
mse = mean_squared_error(y,predicted_score)
rmse = np.sqrt(mse)
r2 = r2_score(y,predicted_score)

# show result 
print("Mean Absolute Error (MAE) :",round(mae,2))
print("Mean Squared Error (MSE) :",round(mse,2))
print("Root Mean Squared Error (RMSE) :",round(rmse,2))
print("R^2 Score (Model Accuracy) :",round(r2,4))
# closer to 1.0 --> better/perfect prediction
# 0.0 --> model does know better then just predcting the average
# < 0.0 --> model is worse then predicting the mean

# Histogram
plt.figure(figsize=(10,6))
plt.hist(data["Final_Score"],bins=30,color="skyblue",edgecolor="black")
plt.title("Distribution Of FINAL EXAM SCORES")
plt.x_label("Final Exam Score")
plt.y_label("Number Of Students")
plt.grid(True)
plt.show()

# scatter - Regression line
# Histogram
plt.figure(figsize=(10,6))
plt.scatter(x,y,color="blue",label = "Actual scores")
plt.plot(x,predicted_score,color="red",label = "Predicted scores(Regression Line)")
plt.title("Model Prediction v/s Actual Score")
plt.x_label("Study Hours Per Week")
plt.y_label("Final Output")
plt.grid(True)
plt.show()


new_hour = 9.5
new_pred = model.predict(new_hour)
print(f"Predicted Final score for {new_hour}hrs is {new_pred}score")