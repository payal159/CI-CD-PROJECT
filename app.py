import numpy as np 
import pandas as pd 

df = pd.read_csv("covid_toy.csv")
print(df.head(2)) 

print(df.isnull().sum()) 
#fill the missing values in the 'fever'column
from sklearn.impute import SimpleImputer 
si = SimpleImputer() 
df['fever'] = si.fit_transform(df[['fever']])

print(df.isnull().sum())

#convert categorical columns to numerical using label encoding
from sklearn.preprocessing import LabelEncoder 
lb=  LabelEncoder() 
df['gender'] = lb.fit_transform(df['gender'])
df['cough'] = lb.fit_transform(df['cough'])
df['city'] = lb.fit_transform(df['city'])
df['has_covid'] = lb.fit_transform(df['has_covid'])

print(df.head(2)) 

# Now we can train a model using the preprocessed data
x = df.drop(columns = ['has_covid'])
y = df['has_covid'] 
from sklearn.model_selection import train_test_split 
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42) 

# Train a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train , y_train) 
y_pred = rf.predict(x_test) 

# Evaluate the model
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test , y_pred))