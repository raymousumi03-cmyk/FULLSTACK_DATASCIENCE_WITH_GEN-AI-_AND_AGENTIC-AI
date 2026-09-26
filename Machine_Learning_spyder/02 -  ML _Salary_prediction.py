import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

ds = pd.read_csv(r"D:\04) Naresh IT\Salary_Data.csv")

#-------------------------------------------------------------------------------------------

x = ds.iloc[:,:-1]
y = ds.iloc[:,-1]

#---------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.2,
                                               train_size=0.8,
                                               random_state=0)

#-----------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

print(regressor) #regressor is a ml model which consider linear regression algorithm
print(regressor.get_params()) #these parameter used to build the model

y_pred = regressor.predict(x_test)
print(y_pred)

comparision = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
print(comparision)

#--------------------------------------------------------------------------------------------

plt.scatter(x_test, y_test, color = 'Red')
plt.plot(x_train, regressor.predict(x_train), color ='blue')
plt.title('Salary of employee based on experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#------------------------------------------------------------------------------------------------------

m_slope = regressor.coef_
print(m_slope)


c_intercept =regressor.intercept_
print(c_intercept)

#-----------------------------------------------------------------------------------------

y_12 = (m_slope*12)+c_intercept
print(y_12)


y_15 = (m_slope*15)+c_intercept
print(y_15)

#---------------------------------------------------------------------------------------

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

#----------------------------------------------------------------------------------------
#statistics for ml regressor model

#Mean---------------------------------------------------------------------------------

ds.mean()

ds['Salary'].mean()

ds['YearsExperience'].mean()

#Median-----------------------------------------------------------------------------

ds['YearsExperience'].median()


ds['Salary'].median()


#Variance(Spread of data around mean)--------------------------------------------------------------------------------

ds.var()

ds['Salary'].var()

ds['YearsExperience'].var()


#Standard deviation--------------------------------------------------------------------------------------

ds.std()

ds['Salary'].std()

ds['YearsExperience'].std()


#coefficient of variation(cv)---------------------------------------------------------------------------------

from scipy.stats import variation
variation(ds.values) # this will give cv of entire df

variation(ds['Salary']) # this will give us cv of that particular column


#correlation

ds.corr()

ds['Salary'].corr(ds['YearsExperience'])


#Skewness------------------------------------------------------------------------------------------

ds.skew()


#Standard Error---------------------------------------------------------------------------------------

ds.sem() # this will give standard error of entire ds


#ANOVA---------------------------------------------------------------------------------------------------

#SSR

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)


#SST

mean_total = np.mean(ds.values)
SST = np.sum((ds.values-mean_total)**2)
print(SST)

#R2

r_square = 1-SSR/SST
print(r_square)

from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)

print(test_mse)
print(train_mse)