#Logistic Regression

#Importing pandas library to read CSV data file
import pandas as pd

#Reading CSV data file in Python
dataset = pd.read_csv('Bank_Data.csv')

#Dividing dataset into X and y
X = dataset.iloc[:, [0,2,4,5]].values
y = dataset.iloc[:, -1].values

#Importing train_test_split from sklearn.model_selection to split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,  y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Importing StandardScaler from sklearn.preprocessing to scale matrix of features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Importing LogisticRegression from sklearn.linear_model to build LogisticRegression classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

predictionResult = (103 + 61) / len(X_test)