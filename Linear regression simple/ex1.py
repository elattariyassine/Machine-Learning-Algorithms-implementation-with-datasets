# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Create Simple Linear Regression

#Importing pandas library to read CSV data file
import pandas as pd

#Importing matplotlib.pyplot library to plot the results
import matplotlib.pyplot as plt

#Reading CSV data file in Python
dataset = pd.read_csv('Study_Hours.csv')

#Dividing dataset into X and y
#take all rows and exclude the last one
X = dataset.iloc[:, :-1].values
#take all rows from column with index 1
y = dataset.iloc[:, 1].values


#Dividing X and y into Training and Testing sets
from sklearn.model_selection import train_test_split
#without random_state data spliting will be randomly, but using that param it will start
#spliting from the first element
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#Importing LinearRegression Library
from sklearn.linear_model import LinearRegression
#Fitting Regression
s_regression = LinearRegression()
s_regression.fit(X_train, y_train)

# Plotting Training  results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, s_regression.predict(X_train), color = 'blue')
plt.title('Study_Hours vs Exam_Score (Training set)')
plt.xlabel('Study_Hours')
plt.ylabel('Exam_Score')
plt.show()

# Plotting Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, s_regression.predict(X_train), color = 'blue')
plt.title('Study_Hours vs Exam_Score (Test set)')
plt.xlabel('Study_Hours')
plt.ylabel('Exam_Score')
plt.show()