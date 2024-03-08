#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:05:33 2024

@author: arnaudcruchaudet
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix
import seaborn as sns

titanic_data = pd.read_csv("titanic.csv", header = 0, sep = ",")
titanic=titanic_data
print (titanic_data.head())

titanic_data["gender"] = np.where(titanic_data["Sex"].str.contains("female"), 1, 0)

titanic_data["lnAge"]=np.log(titanic_data.Age)
titanic_data["lnFare"]=np.log(titanic_data.Fare+1)
titanic_B=titanic_data.dropna(subset=["Age"])


#OLS estimator
X=titanic_B[["Pclass", "lnFare", "lnAge", "gender", "SibSp", "Parch"]]
X = sm.add_constant(X)
Y=titanic_B[["Survived"]]
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())


ypred=results.predict(X)
print(ypred)
titanic_B.loc[:,"ypred"]=ypred
#Generate the confusion matrix for OLS
threshold = 0.5
# transforms the list of booleans into an int with 1 = True, 0 = False
predicted_class = (ypred > threshold).astype(int)
print(predicted_class)
titanic_B.loc[:,"Bpred"]=predicted_class
Z=titanic_B["Bpred"]
c1 = confusion_matrix(Y,Z)
print(c1)

sns.heatmap(c1, annot=True,  fmt='', cmap='Blues')
sns.heatmap(c1/np.sum(c1), annot=True,  fmt='.2%', cmap='Blues')






#Maximul Likelihood (logit model)
logistic_model = sm.GLM(Y, X, family=sm.families.Binomial())
results1 = logistic_model.fit()
print(results1.summary())
#Obtain the same result with this command
logit_mod = sm.Logit(Y, X)
results2 = logit_mod.fit()
print(results2.summary())
#Generating odds ratios
print(np.exp(results2.params))
# We don't interpret this coefficients as the same
# We look whether there are superior than 1 or not
# Because there isn't negative coeff


print(titanic_B.describe("lnAge"))
K=titanic_B["Age"]
print(K.describe())


yplogit=results2.predict(X)
print(yplogit)
titanic_B.loc[:,"yplogit"]=yplogit #never have negative forecast
#Generate the confusion matrix
threshold = 0.5
# transforms the list of booleans into an int with 1 = True, 0 = False
predicted_class2 = (yplogit > threshold).astype(int)
print(predicted_class2)
titanic_B.loc[:,"Bpred2"]=predicted_class2
Z=titanic_B["Bpred2"]
c2 = confusion_matrix(Y,Z)
print(c2)

sns.heatmap(c2, annot=True,  fmt='', cmap='Blues')

#Compare logit with OLS
sns.heatmap(c2/np.sum(c2), annot=True,  fmt='.2%', cmap='Blues')
sns.heatmap(c1/np.sum(c1), annot=True,  fmt='.2%', cmap='Blues')

#Probit model use Normal CDF instead of Logistic cdf
probit_mod = sm.Probit(Y, X)
rprob = probit_mod.fit()
print(rprob.summary())
yprob=rprob.predict(X)
print(yprob)
titanic_B.loc[:,"yprob"]=yprob
#Generate the confusion matrix
threshold = 0.5
# transforms the list of booleans into an int with 1 = True, 0 = False
predicted_class3 = (yprob > threshold).astype(int)
print(predicted_class3)
titanic_B.loc[:,"Bpred3"]=predicted_class3
Z=titanic_B["Bpred3"]
c3 = confusion_matrix(Y,Z)
print(c3)


sns.heatmap(c3, annot=True,  fmt='', cmap='Blues')


#Compare Probit with logit and OLS
sns.heatmap(c3/np.sum(c3), annot=True,  fmt='.2%', cmap='Blues')
sns.heatmap(c2/np.sum(c2), annot=True,  fmt='.2%', cmap='Blues')
sns.heatmap(c1/np.sum(c1), annot=True,  fmt='.2%', cmap='Blues')


#Poisson model

poisson_model =sm.GLM(Y,X,family=sm.families.Poisson())
results = poisson_model.fit()
print(results.summary())
#or
poisson_mod = sm.Poisson(Y,X)
rpois = poisson_mod.fit()
print(rpois.summary())
ypois=rpois.predict(X)
print(ypois)
titanic_B.loc[:,"ypois"]=ypois
#Generate the confusion matrix
threshold = 0.5
# transforms the list of booleans into an int with 1 = True, 0 = False
predicted_class4 = (ypois > threshold).astype(int)
print(predicted_class4)
titanic_B.loc[:,"Bpred4"]=predicted_class4
Z=titanic_B["Bpred4"]
c4 = confusion_matrix(Y,Z)
print(c4)
K1=titanic_B["Bpred4"]
K2=titanic_B["Survived"]


sns.heatmap(c4/np.sum(c4), annot=True,  fmt='.2%', cmap='Blues')
sns.heatmap(c2/np.sum(c2), annot=True,  fmt='.2%', cmap='Blues')
# Type I error and Type II errror or not the same importance according the context

pd.crosstab(K2, K1, rownames=['True'], colnames=['Predicted'], margins=True)


print(c1, c2, c3, c4)

#Multinomial
Mlogit = smf.mnlogit("Pclass ~ gender + Age+Fare", titanic_B).fit()
Mlogit.summary()
ymlogit=Mlogit.predict(titanic_B[['gender','Age', 'Fare']])
print(ymlogit)
Titanic_new= pd.concat([titanic_B, ymlogit], axis=1, join="inner")
Titanic_new["Pred2"] =Titanic_new[[0,1,2]].idxmax(axis=1)
Titanic_new["Pred"] = Titanic_new["Pred2"]+1
Z=Titanic_new["Pred"]
Y=Titanic_new["Pclass"]
c4 = confusion_matrix(Y,Z)
print(c4)