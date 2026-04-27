'''

Linear Regression

1. find the pattern from the old data 
2. straight line
3.predct from extending the line

syntax

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
model.predict([[value]])

'''
from sklearn.linear_model import LinearRegression

x = [[1],[2],[3],[4],[5]] 
y = [40,50,65,75,90]

model = LinearRegression()
model.fit(x,y)

hours = float(input("Enter how many hrs you studied"))
predicted_marks = model.predict([[hours]])

print("Based on (hours) hrs you may score around [predicted_marks] ")