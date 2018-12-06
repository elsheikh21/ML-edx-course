import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

'''
DataSet:
MODELYEAR e.g. 2014
MAKE e.g. Acura
MODEL e.g. ILX
VEHICLE CLASS e.g. SUV
ENGINE SIZE e.g. 4.7
CYLINDERS e.g 6
TRANSMISSION e.g. A6
FUEL CONSUMPTION in CITY(L/100 km) e.g. 9.9
FUEL CONSUMPTION in HWY (L/100 km) e.g. 8.9
FUEL CONSUMPTION COMB (L/100 km) e.g. 9.2
CO2 EMISSIONS (g/km) e.g. 182 --> low --> 0
'''
# Reading the data in
df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()

# Data Exploration

# Lets select some features to explore more
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
          'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

# lets plot each of these features vs the Emission,
# to see how linear is their relation
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Creating train and test dataset
# split our dataset into train and test sets,
# 80% of the entire data for training, and the 20% for testing
# create a mask to select random rows using np.random.rand() function
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# multiple variables that predict the Co2emission. When more than one
# independent variable is present, the process is called
# multiple linear regression. For example, predicting co2emission
# using FUELCONSUMPTION_COMB, EngineSize and Cylinders of cars.
# The good thing here is that Multiple linear regression is the
# extension of simple linear regression model.

# Coefficient and Intercept , are the parameters of the fit line.
# Given that it is a multiple linear regression, with 3 parameters,
# and knowing that the parameters are the intercept and
# coefficients of hyperplane, sklearn can estimate them from our data.
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
# The coefficients
print('Coefficients: ', regr.coef_)

# Scikit-learn uses plain Ordinary Least Squares method to solve this problem.
# Ordinary Least Squares(OLS)
# OLS is a for estimating the unknown parameters in a linear regression model.
# OLS chooses the parameters of a linear function of a set of explanatory
# variables by minimizing the sum of the squares of the differences between
# the target dependent variable and those predicted by the linear function.
# In other words, it tries to minimizes the sum of squared errors (SSE) or
# mean squared error (MSE) between the target variable (y) and our predicted
# output ( ŷ y^ ) over all samples in the dataset.

# OLS can find the best parameters using of the following methods:

# - Solving the model parameters analytically using closed-form equations
# - Using an optimization algorithm (Gradient Descent,
#       Stochastic Gradient Descent, Newton’s Method, etc.)

y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

# explained variance regression score:
# If  ŷ y ^ is the estimated target output, y the corresponding(correct)
# target output, and Var is Variance, the square of the standard deviation,
# then the explained variance is estimated as follow:

# 𝚎𝚡𝚙𝚕𝚊𝚒𝚗𝚎𝚍𝚅𝚊𝚛𝚒𝚊𝚗𝚌𝚎(y, ŷ) =
# 1−Var{y−ŷ}/Var{y}
# The best possible score is 1.0, lower values are worse.
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS',
                         'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_CITY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
# The coefficients
print('\nCoefficients: ', regr.coef_)

y_hat = regr.predict(
    test[['ENGINESIZE', 'CYLINDERS',
          'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_CITY']])
x = np.asanyarray(
    test[['ENGINESIZE', 'CYLINDERS',
          'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_CITY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))


print('Variance score: %.2f' % regr.score(x, y))
