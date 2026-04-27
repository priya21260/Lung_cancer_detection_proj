'''

ex --> 

problem                     predict                   classification label

email                       spam /                     yes / no (2 label)
filter                       not spam

hospitality                 bimar ho / nahi ho          yes / no (2 label)
                           

'''

# Logistic Regression 

from sklearn. linear_model import LogisticRegression

x = [[1],[2],[3],[4],[5]]
y = [0,0,1,1,1]  # result 0-> fail 1-> pass

model = LogisticRegression()
model.fit(x,y)
hours = float(input("Enter how many hrs you studied"))
result = model.predict([[hours]]) [0]

if result == 1 :
    print(f'Based on your {hours} ,you are likely to PASS')
else :
    print(f'Based on {hours}hrs,you are likely to FAIL')

# KNN --> K - Nearest Neighbous

'''

K - Nearest Neighbous --> slow for big data
alway choose odd value

KNN 

weight(gm)               size(cm)               fruit
180                        7                    Apple
200                       7.5                   Apple
250                        8                    Apple
300                       8.5                   Orange
330                        9                    Orange
360                        9.5                  Orange

'''

from sklearn.neighbors import KNeighborsClassifier
x = [
      [180,7],
      [200,7.5],
      [250,8],
      [300,8.5],
      [330,9],
      [360,9.5]
    ]
# 0--> Apple ,1--> Orange
y = [0,0,0,1,1,1]
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)
weight = float(input("Enter the weight in grams :"))
size = float(input("Enter the size in cm :"))
prediction = model.predict([[weight,size]]) [0]
# bina [0] --> it returns [1] (bag me 1 ) but we nned only 1 (if it is orange)
if prediction == 0 :
    print("This is likely an Apple")
else :
    print("This is likely an Orange")

# Decision Tree

'''
Decision Tree




'''

from sklearn.tree import DecisionTreeClassifier
x = [
    [7,2], # Apple
    [8,3], # Apple
    [9,8], # Orange
    [10,9] # Orange
] 

y = [0,0,1,1] # 0-->Apple , 1--> Orange
model = DecisionTreeClassifier()
model.fit(x,y) # only once
size = float(input("Enter the fruit size in cm:"))
shade = float(input('Enter the shade (1-10):'))
result = model.predict([[size , shade]])[0] # more then once

if result == 0 :
    print("This is likely an Apple")
else :
    print("This is likely an Orange")
