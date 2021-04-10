#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:49:12 2021

@author: yassine
"""

# Multiple Linear Regression

#Importing pandas library to read CSV data file
import pandas as pd

#Importing numpy to perfom math on arrays
import numpy as np

#Importing statsmodels.formula to fit the regression model
import statsmodels.formula.api as sm


#Reading CSV data file in Python
dataset = pd.read_csv('Companies spends and profits.csv')


#Dividing dataset into X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data to create the dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#labelencoder = LabelEncoder()
#transforming cities names into numbers
#X[:, 0] = labelencoder.fit_transform(X[:, 0])

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])],     remainder='passthrough')
X=np.array(columnTransformer.fit_transform(X),dtype=np.str)

#creating an instance and chaging cities from numbers (1,2,3...) to only 0 and 1 
#onehotencoder = OneHotEncoder(categorical_features = [0])
#override those numbers with others columns of 0 and 1
#X = onehotencoder.fit_transform(X).toarray()

#Adding intercept column to X
X = np.append(arr = np.ones((71, 1)).astype(int), values = X, axis = 1)

# Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting the first regression model
X_fitted = X_train[:,:]
reg_1 = sm.OLS(endog = y_train, exog = X_fitted).fit()
reg_1.summary()

#Fitting the second regression model
X_fitted = X_train[:, [0,1,3,4,5,6,7]]
reg_2 = sm.OLS(endog = y_train, exog = X_fitted).fit()
reg_2.summary()

#Fitting the second regression model
X_fitted = X_train[:, [0,3,4,5,6,7]]
reg_3 = sm.OLS(endog = y_train, exog = X_fitted).fit()
reg_3.summary()

#Fitting the second regression model
X_fitted = X_train[:, [0,3,4,5,7]]
reg_4 = sm.OLS(endog = y_train, exog = X_fitted).fit()
reg_4.summary()

#Fitting the second regression model
X_fitted = X_train[:, [0,3,4,5]]
reg_5 = sm.OLS(endog = y_train, exog = X_fitted).fit()
reg_5.summary()

#Fitting the second regression model
X_fitted = X_train[:, [0,3,5]]
reg_6 = sm.OLS(endog = y_train, exog = X_fitted).fit()
reg_6.summary()

#Fitting the second regression model
X_fitted = X_train[:, [0,5]]
reg_7 = sm.OLS(endog = y_train, exog = X_fitted).fit()
reg_7.summary()

X_for_test = X_test[:, [0,5]]

y_prdicted = reg_7.predict(X_for_test)