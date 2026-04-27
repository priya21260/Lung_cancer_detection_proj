import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
import joblib


url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
data = pd.read_csv(url)
print(data.head())

print(data.describe())
print(data.info())
print(data.shape)

# handling missing data
data.fillna(data.median(numeric_only=True),inplace=True)

# encoding categorical data --> one hot encoding
data = pd.get_dummies(data,columns=["ocean_proximity"])

x = data.drop("median_house_value",axis=1)
y = data["median_house_value"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("model",RandomForestRegressor(n_estimators=100))
])

pipeline.fit(x_train,y_train)

y_pred = pipeline.predict(x_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("MAE:",mae)
print("MSE:",mse)
print("r2_score:",r2) 

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Price")
plt.ylabel("predicted Price")
plt.title("Actual vs Predicted")
plt.show()

# HYPERPARAMETER TUNING

param_grid = {
    "model__n_estimators":[50,100],
    "model__max_depth":[None,10]
}

grid = GridSearchCV(pipeline,param_grid,cv=3)
grid.fit(x_train,y_train)
print(grid.best_params_)

joblib.dump(grid.best_estimator_,"house_model.pkl")
model = joblib.load("house_model.pkl")
model.predict(x_test[:5])
