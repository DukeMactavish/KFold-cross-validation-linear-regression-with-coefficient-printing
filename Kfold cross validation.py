#by DukeMactavish
#kfold cross validation for linear regression using sklearn library
#Will print the coeffcients and intercept
# along with metrics mean absolute error, mean squared error, root mean square error and r^2 score for each iteration
#provided code is for two different dependent variables and hence two different regressions, delete second one if not needed

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
#Data importing modify source and header names according to to necessity
df=pd.read_csv("D:/Test_set.csv") #reading from csv change source as needed
df.head()
Y1=df['OT'] #first dependent variable
Y2=df['OD'] #second dependent variable
X=df[['LCV','HV','Xg','Vb','DeltaV']] #independent variables

regressor=LinearRegression() #Linear regression

#NUMBER OF FOLDS REQUIRED MAYBE CONTROLLED HERE USING K
k=10 #change k according to number of folds(Default=10)

kf=KFold(n_splits=k, random_state=1, shuffle=True)  #Splitting the folds k times

#for first dependent variable
print("------------FOR OT----------------")

#Iteration control
i=1
#lists to store metrics
mae=[]
mse=[]
rmse=[]
r2=[]

for train_index, test_index in kf.split(X):
    print('ITERATION: ',i)
    i=i+1

    #splitting
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y1.iloc[train_index], Y1.iloc[test_index]

    #fitting
    regressor.fit(X_train, y_train)

    # To retrieve the intercept:
    print('Intercept: ',regressor.intercept_)
    # For printing the coefficients:
    print("Coefficients: ")
    print(regressor.coef_)

    #predicting
    y_pred = regressor.predict(X_test)

    #plotting if necessary
    # plt.scatter(X_test['Xg'],y_test,label="Tester")
    # plt.scatter(X_test['Xg'],y_pred,label="Predicted")
    # plt.legend()
    # plt.show()

    #Calculating metrics
    ma= metrics.mean_absolute_error(y_test, y_pred)
    ms= metrics.mean_squared_error(y_test, y_pred)
    mrm= np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mr2= metrics.r2_score(y_test, y_pred)
    #appending metrics
    mae.append(ma)
    mse.append(ms)
    rmse.append(mrm)
    r2.append(mr2)
    #printing metrics for fit
    print('Mean Absolute Error:',ma)
    print('Mean Squared Error:',ms)
    print('Root Mean Squared Error:',mrm)
    print('R^2 score:',mr2)

#Printing mean values of metrics
print("AVERAGE METRICS:")
print('Average Mean Absolute Error:',np.mean(np.array(mae)))
print('Average Mean Squared Error:', np.mean(np.array(mse)))
print('Average Root Mean Squared Error:', np.mean(np.array(rmse)))
print('Average R^2 score:', np.mean(np.array(r2)))


#For second dependent variable, can be deleted if only one variable is to be fitted for
print("------------FOR OD----------------")

#iteration control
i=1

for train_index, test_index in kf.split(X):
    print('ITERATION: ',i)
    i=i+1
    #splitting
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y2.iloc[train_index], Y2.iloc[test_index]

    #fitting
    regressor.fit(X_train, y_train)

    # To retrieve the intercept:
    print('Intercept: ',regressor.intercept_)
    # For retrieving the coefficients:
    print("Coefficients: ")
    print(regressor.coef_)

    #predicting
    y_pred = regressor.predict(X_test)

    #plotting if necessary
    # plt.scatter(X_test['Xg'],y_test,label="Tester")
    # plt.scatter(X_test['Xg'],y_pred,label="Predicted")
    # plt.legend()
    # plt.show()

    #Calculating metrics
    ma= metrics.mean_absolute_error(y_test, y_pred)
    ms= metrics.mean_squared_error(y_test, y_pred)
    mrm= np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mr2= metrics.r2_score(y_test, y_pred)
    #appending metrics
    mae.append(ma)
    mse.append(ms)
    rmse.append(mrm)
    r2.append(mr2)
    #printing metrics for fit
    print('Mean Absolute Error:',ma)
    print('Mean Squared Error:',ms)
    print('Root Mean Squared Error:',mrm)
    print('R^2 score:',mr2)

#Printing mean values of metrics
print("AVERAGE METRICS:")
print('Average Mean Absolute Error:',np.mean(np.array(mae)))
print('Average Mean Squared Error:', np.mean(np.array(mse)))
print('Average Root Mean Squared Error:', np.mean(np.array(rmse)))
print('Average R^2 score:', np.mean(np.array(r2)))
