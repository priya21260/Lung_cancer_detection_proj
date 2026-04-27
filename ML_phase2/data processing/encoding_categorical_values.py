'''

from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("sample_data.csv")
df_label = df.copy()
le = LabelEncoder()
df_label['Gender_Encoded'] = le.fit_transform(df_label['Gender']) 
df_label['Passed_Encoded'] = le.fit_transform(df_label['Passed'])

print("\nLabel Encoded Data")
print(df_label[['Name','Gender','Gender_Encoded','Passed','Passed_Encoded']])

df_encoded = pd.get_dummies{df_label,columns = ['city']}
print("\nOn Hot Encoded Data [City]")
print(df_encoded)

'''