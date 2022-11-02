# 1. Apply Linear Regression to the provided dataset using underlying steps.
#      a. Import the given “Salary_Data.csv”
#      b. Split the data in train_test partitions, such that 1/3 of the data is reserved as test subset.
#      c. Train and predict the model.
#      d. Calculate the mean_squared error
#      e. Visualize both train and test data using scatter plot.

# Simple Linear Regression
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the datasets
salary_dataset = pd.read_csv('C:/Users/Kiran Kumar Kongari/PycharmProjects/ML-Assignment-4/Datasets/Salary_Data.csv')

print(salary_dataset)

X = salary_dataset.iloc[:, :-1].values
Y = salary_dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Fitting Simple Linear Regression to the training set
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# Predicting the Test set result ￼
Y_Pred = regressor.predict(X_Test)

# calculate mean square error
mse = mean_squared_error(Y_Test,Y_Pred)
print(f"\nMean Square Error = {mse}")

# Visualising the Training set results
plt.scatter(X_Train, Y_Train,  color='red')
plt.title('Training Dataset (Salary vs Years Of Experience)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_Test, Y_Test,  color='red')
plt.title('Test Dataset (Salary vs Years Of Experience)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# plotting the actual and predicted values
compare = [i for i in range(1, len(Y_Test)+1, 1)]
plt.plot(compare, Y_Test, color='green', linestyle='-')
plt.plot(compare, Y_Pred, color='red', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Salary')
plt.title('Actual salary value(green line) vs Predicted Salary values(red line)')
plt.show()

# Plotting the Final Output i.e., the test data and predicted data
plt.scatter(X_Test, Y_Test,  color='red')
plt.plot(X_Test, Y_Pred, color='black', linewidth=3)
plt.title('Salary vs Years Of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

