# Create Polynomial Regression Model

#Importing pandas library to read CSV data file
import pandas as pd

#Importing matplotlib to plot the results
import matplotlib.pyplot as plt

#Reading CSV data file in Python
dataset = pd.read_csv('Reward_system.csv')

#Dividing dataset into X and y
#X = dataset.iloc[:, [0]].values
#same as top, get from 0 to 1, 1 excluded
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

#Importing LinearRegression from sklearn.linear_model to fit linear regression model
from sklearn.linear_model import LinearRegression

# Fitting Linear Regression to training data
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Plotting Linear Regression results
plt.scatter(X, y, color = 'black')
plt.plot(X, linear_reg.predict(X), color = 'red')
plt.title('Reward System (Linear Regression)')
plt.xlabel('Hours')
plt.ylabel('Points')
plt.show()

#Importing PolynomialFeatures from sklearn.preprocessing to fit polynomial regression model
from sklearn.preprocessing import PolynomialFeatures

# Fitting Polynomial Regression to training data
polynomial_reg = PolynomialFeatures(degree = 3)
X_polynomial = polynomial_reg.fit_transform(X)
polynomial_reg.fit(X_polynomial, y)
linear_polynomial = LinearRegression()
linear_polynomial.fit(X_polynomial, y)

# Plotting Polynomial Regression results
plt.scatter(X, y, color = 'black')
plt.plot(X, linear_polynomial.predict(polynomial_reg.fit_transform(X)), color = 'red')
plt.title('Reward System (Linear Regression)')
plt.xlabel('Hours')
plt.ylabel('Points')
plt.show()

# Predicting a y value using Linear Regression
y_pred_by_simple_linear =linear_reg.predict([[90]])

# Predicting a y value using Polynomial Regression
y_pred_by_polynomial_linear = linear_polynomial.predict(polynomial_reg.fit_transform([[90]]))