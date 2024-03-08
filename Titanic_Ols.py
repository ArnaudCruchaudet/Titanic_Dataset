#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:54:35 2024

@author: arnaudcruchaudet
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
import seaborn as sns



titanic_data = pd.read_csv("titanic.csv", header = 0, sep = ",")
titanic=titanic_data
print (titanic_data.head())

#Information on the variables
print(type(titanic_data))
print(titanic_data.info())

#discretization of qualitative variable
titanic_data["gender"] = np.where(titanic_data["Sex"].str.contains("female"), 1, 0)



titanic_data["Age2"]=titanic_data.Age**2
titanic_data["Fare2"]=titanic_data.Fare**2
titanic_data["lnAge"]=np.log(titanic_data.Age)
titanic_data["lnFare"]=np.log(titanic_data.Fare+1) # log(0) doesn't exist, this is why we had +1


titanic_B=titanic_data.dropna(subset=["Age"]) # drop all null cases



X=titanic_B[["Pclass", "Fare", "Age", "gender", "SibSp", "Parch"]]
X = sm.add_constant(X) # constant whatever covariable, is true for all varibale (column of 1)
Y=titanic_B[["Survived"]] # varibale we want to predict
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())
# if you are female, you are 48,85% chance more than human than male


X=titanic_B[["Pclass", "Fare", "Age", "gender", "SibSp", "Parch", "Age2", "Fare2"]]
X = sm.add_constant(X)
Y=titanic_B[["Survived"]]
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())



# Here we want to see the impact of fare
X=titanic_B[["Fare", "Age", "gender", "SibSp", "Parch"]]
X = sm.add_constant(X)
Y=titanic_B[["Survived"]]
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())



X=titanic_B[["Pclass", "lnFare", "lnAge", "gender", "SibSp", "Parch"]]
X = sm.add_constant(X)
Y=titanic_B[["Survived"]]
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())


# Generate prediction
ypred=results.predict(X)
print(ypred)
titanic_B.loc[:,"Ypred"]=ypred
#Generate the confusion matrix for OLS
threshold = 0.5
# transforms the list of booleans into an int with 1 = True, 0 = False
predicted_class = (ypred > threshold).astype(int)
print(predicted_class)
titanic_B.loc[:,"Bpred"]=predicted_class
Z=titanic_B["Bpred"]
results = confusion_matrix(Y,Z)
print(results)

y_pred = pd.Series(titanic_B["Bpred"])
y_true = pd.Series(titanic_B["Survived"])
pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


sns.heatmap(results, annot=True,  fmt='', cmap='Blues')
sns.heatmap(results/np.sum(results), annot=True,  fmt='.2%', cmap='Blues')