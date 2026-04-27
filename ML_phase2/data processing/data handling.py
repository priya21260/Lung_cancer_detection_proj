import pandas as pd

data = {
    'Name' : ['pavan','kapil','lalit','ishan','om'] , 
    'Age' : [25,None,44,23,None],
    'Salary' : [50000,60000,70000,None,None] 
}

df = pd.DataFrame(data)
print("original DataFrame ")
print(df)

print (df.isnull().sum())
df_drop = df.dropna()
print(df_drop)

# df['Age'].fillna(df['Age'].mean(),inplace = True)
# df['Salary'].fillna(df['Salary'].mean(),inplace = True)

df.fillna(df.mean(numeric_only=True), inplace=True)
print(df)

# Numeric data --> fill with mean / median
# categorical data --> fill with mode / place holder --> unknown

print(df.isnull().mean()*100) # percent of data missing

