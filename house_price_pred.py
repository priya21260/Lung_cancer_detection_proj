import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"


data = pd.read_csv(url)

print(data.head())

print(data.info())
print(data.describe())
print(data.shape)
print(data.isnull().sum())

#Affects of each column on House Price
# using the correlation matrix --> 
# tells how strongly each numeric feature is realated to house price
corr = data.corr(numeric_only=True)
print(corr["median_house_value"].sort_values(ascending=True))

# filling missing values
data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())

# trying to gets diff type of places --> decide label/hot code encoding
print(data["ocean_proximity"].value_counts())

# for last column --> using one hot encoding 
data = pd.get_dummies(data,columns=['ocean_proximity'],drop_first=True)
print(data)


x = data.drop("median_house_value",axis=1)
y = data["median_house_value"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("R2 Score:",r2)