# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:28:02 2020

@author: kingslayer
"""

#MY OWN MULTIPLE LINEAR REGRESSION MODEL

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset 
dataset=pd.read_csv(r"Data.csv")

#creating matrix of features
X=dataset.iloc[:,:-1].values
#creating dependant variable vector
y=dataset.iloc[:,-1].values

#encoding the categorical data 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
X=X[:,1:]

#Spliting into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting
y_pred=regressor.predict(X_test)

#building optimal model
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


#Spliting into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_opt,y,test_size=0.2,random_state=0)

#model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting
y_pred=regressor.predict(X_test)

#Evaluating
from sklearn import metrics
print(f"Mean Squared Error:{metrics.mean_squared_error(y_test,y_pred)}")