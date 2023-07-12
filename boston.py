# Example: Predict Price of House, given #Rooms
# Need: Regression Algorithm, predicts _continous values_ (like price)
#
# * Uses Example Data available in Sklearn (Boston Housing Data) csv file
#
# Task: "I want 5 rooms, how much does a house cost in Boston?"
#
# Steps:  
# 1. Load Data
# 2. Algorithm is trained on data.
# 3. Algorithm predicts price, given #Rooms
#


# Load sklearn functionality
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import pandas


dataset=pandas.read_csv('boston_house.csv')



x=dataset["RM"].values.reshape(-1, 1)
y=dataset["MEDV"]
    

    
# Split train/test set
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.50)

# # 2. Algorithm is trained on data.
lm = LinearRegression()
lm.fit(x_train, y_train)

# # 3. Algorithm predicts price, given #Rooms
y_pred = lm.predict(x_test)

# # Create a plot showing predictions
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='red', linewidth=3)

plt.xlabel("Number of Rooms")
plt.ylabel("Predicted prices: $\hat{y}_i$")
plt.title("Prices vs Predicted prices")
plt.show()

# Make some prediction
eight=lm.predict([[8]])
five=lm.predict([[5]])
print("The price of 8 Rooms is %.2f"%eight)
print("The price of 5 Rooms is %.2f"%five)