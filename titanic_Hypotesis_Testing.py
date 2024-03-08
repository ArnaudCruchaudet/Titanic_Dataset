#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:14:13 2024

@author: arnaudcruchaudet
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

titanic_data = pd.read_csv("titanic.csv", header = 0, sep = ",")
titanic=titanic_data
print (titanic_data.head())



#Information on the variables
print(type(titanic_data))
print(titanic_data.info())
#One sample mean Test (two sided test)
print(titanic_data.Age.mean())
print(stats.ttest_1samp(titanic_data.Age, 30, nan_policy="omit"))
#accept the null
print(stats.ttest_1samp(titanic_data.Age, 35, nan_policy="omit"))
#reject the null



print(titanic_data.Fare.mean())
print(stats.ttest_1samp(titanic_data.Fare, 30, nan_policy="omit"))
#accept the null
print(stats.ttest_1samp(titanic_data.Fare, 32, nan_policy="omit"))
#reject the null



# Equal_Var=False or True!!!!!!!
#two sample mean test (two sided)
print(stats.ttest_ind(titanic_data["Age"], titanic_data["Fare"], nan_policy="omit")) # 1 way
print(stats.ttest_ind(titanic_data.Age, titanic_data.Fare, nan_policy="omit")) # The other way
#Conclusion: we accept the null hypothesis
#Before concluding seriously, we should check the underlying hypthesis of equal variance in the two samples
print(titanic_data.Age.std(), titanic_data.Fare.std())
#this hypiothesis is too strong, we have to run the Welch's t-test for unequal variances/ sample sizes
print(stats.ttest_ind(titanic_data.Age, titanic_data.Fare, nan_policy="omit", equal_var=False))
#we still accept the null of equal means under heterogeneous variance



#testing mean of Age between female and male
titanic_data["F_Age"]= np.where(titanic_data["Sex"]=="female",titanic_data["Age"], float("NaN"))
titanic_data["M_Age"]= np.where(titanic_data["Sex"]=="male",titanic_data["Age"], float("NaN"))
print(stats.ttest_ind(titanic_data["F_Age"], titanic_data["M_Age"], nan_policy="omit"))
#We reject the null
print(titanic_data.F_Age.mean(), titanic_data.M_Age.mean())
#Before concluding seriously, we should check the underlying hypthesis of equal variance in the two samples
print(titanic_data.F_Age.std(), titanic_data.M_Age.std())
#Variance are very similar
print(stats.ttest_ind(titanic_data["M_Age"], titanic_data["F_Age"], nan_policy="omit", equal_var=False))



# Unilateral test
stats.t.ppf([0.05, 0.95], 712)
# We run bilateral test just using a unilateral test. Here we have 1.64 but
# we got 2.52 so its far from 1.64 so we reject the null 
# We comparate the t value with threshold value 
# Above I reject the null, below I accept the null
# THE ORDER IS VERY IMPORTANTE, MALE(FIRST) OVER FEMALE(SECOND)

# Second try
titanic_data["F_Fare"]= np.where(titanic_data["Sex"]=="female",titanic_data["Fare"], float("NaN"))
titanic_data["M_Fare"]= np.where(titanic_data["Sex"]=="male",titanic_data["Fare"], float("NaN"))
print(stats.ttest_ind(titanic_data["M_Fare"], titanic_data["F_Fare"], nan_policy = "omit"))
# We accept the null

# Third try
titanic_data["F_Pclass"]= np.where(titanic_data["Sex"]=="female",titanic_data["Pclass"], float("NaN"))
titanic_data["M_Pclass"]= np.where(titanic_data["Sex"]=="male",titanic_data["Pclass"], float("NaN"))
print(stats.ttest_ind(titanic_data["F_Pclass"], titanic_data["M_Pclass"], nan_policy = "omit"))



#Normality Tests

#Jarque-Bera Test (only if more than 2000 observations -- but works)
print(stats.jarque_bera(titanic_data.Fare))
print(stats.jarque_bera(titanic_data.Age))

titanic=titanic_data.dropna(subset=["Age"]) #dropna, remove the missing value
print(stats.jarque_bera(titanic.Age))


x=np.random.normal(0, 1, 10000)
print(stats.jarque_bera(x))
#here we accept the null
y1=np.random.normal(10, 4, 10000)
print(stats.jarque_bera(y1))
#here we accept the null
y2=np.random.standard_t(4,10000)
print(stats.jarque_bera(y2))
#here we reject the null
y3=np.random.standard_t(20,10000)
print(stats.jarque_bera(y3))
#here we reject the null
y4=np.random.logistic(0,1,10000)
print(stats.jarque_bera(y4))
#here we reject the null
y5=np.random.exponential(4,10000)
print(stats.jarque_bera(y5))
#here we reject the null


#Shapiro-Wilk (Data <5000 observations)
print(stats.shapiro(titanic_data.Fare))
#We reject the null
#Try with Age Data
print(stats.shapiro(titanic_data.Age))
titanic=titanic_data.dropna(subset=["Age"])
print(stats.shapiro(titanic.Age))
#We reject the null