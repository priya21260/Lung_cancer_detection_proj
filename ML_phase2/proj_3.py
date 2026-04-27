import pandas as pd
import numpy as np
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# loading the dataset
df = pd.read_csv("amazon_reviews.csv")
print("First 5 rows")
print(df.head())

# preprocessing text
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    return text

df['review'] = df['review'].apply(clean_text)

# label encoding 
df['sentiment'] = df['sentiment'].map({'negative':0, 'positive':1})

# train / val / test split (70/15/15)
x = df['review']
y = df['sentiment']

x_train , x_temp , y_train , y_temp = train_test_split(x,y,test_size=0.3,random_state=42)
x_val , x_test , y_val , y_test = train_test_split(x_temp,y_temp,test_size=0.5,random_state=42)

# TF-IDF vectorization

vectorizer = TfidfVectorizer(max_features=10000)

x_train_vec = vectorizer.fit_transform(x_train)
x_val_vec = vectorizer.transform(x_val)
x_test_vec = vectorizer.transform(x_test)

# train model 
model = LogisticRegression(max_iter=1000)
model.fit(x_train_vec,y_train)

# validatin check 
val_pred = model.predict(x_val_vec)
print("\nValidation Accuracy:",accuracy_score(y_val,val_pred))

# Final Testing
y_pred = model.predict(x_test_vec)
print("nTest Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report :\n",classification_report(y_test,y_pred))
print("\nConfusion matrix:\n",confusion_matrix(y_test,y_pred))

# save model & vectorizer
joblib.dump(model,"sentimental_model.pkl")
joblib.dump(vectorizer,"tfidf_vectorizer.pkl")

print("\nModule and vectorizer saved successfully!")

