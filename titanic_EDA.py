#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:11:02 2024

@author: arnaudcruchaudet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

titanic_data = pd.read_csv("titanic.csv", header = 0, sep = ",")



print (titanic_data.head())

#Information on the variables
print(type(titanic_data))
print(titanic_data.info())


print(titanic_data.count()) 

print(titanic_data.describe()) 
print(list(titanic_data.columns))
#To reduce the number of decimals
print(titanic_data.describe(include='all').round(2))
print(titanic_data.describe().round(2))

#Générer Skewness & Kurtosis
X_quanti=titanic_data[["Age","Fare"]]
a=pd.Series(list(X_quanti.columns)) # series = vectors as list
print(a)
b=pd.Series(X_quanti.skew())
c=pd.Series(X_quanti.kurtosis())
d=pd.Series(c+3)
SK= pd.DataFrame(columns = [a, b, c, d])
SK= pd.DataFrame({"Skewness": pd.Series(b, index=a), "Kurtosis excess": pd.Series(c, index=a),"Kurtosis": pd.Series(d, index=a)})
# change the column name the SK
print(SK)
#verifying that this table is ok
print(stats.skew(titanic_data.Age, axis=0, bias=False, nan_policy='omit'))
print(stats.kurtosis(titanic_data.Age, fisher=False, axis=0, bias=False, nan_policy='omit'))
# stats produce true kurtosis and not pandas



#Data Visualization

#Continuous variables
print(titanic_data["Fare"].plot(kind="hist"))
print(titanic_data["Fare"].plot(kind="hist", bins=20))

print(titanic_data["Age"].plot(kind="hist"))


#Binary variable
print(titanic_data["Sex"].value_counts().plot(kind="bar"))
print(titanic_data["Sex"].value_counts(normalize=True).plot(kind="bar")) # present in term of frequency

#Categorical variable
print(titanic_data["Pclass"].value_counts(normalize=True).plot(kind="bar"))
print(titanic_data["Pclass"].value_counts(normalize=True).plot(kind="pie", autopct='%1.1f%%', title="Passenger class distribution"))


#Dealing with missing data
print(titanic_data.isnull().sum())
#1st solution: drop missing value
titanic_data.dropna(subset=['Embarked'], inplace=True)
print(titanic_data.info())
#create a quantitative variable from a categorical (non ordered)
#We convert a column to a category, then use those category values for your label encoding:
titanic_data["Embarked"] = titanic_data["Embarked"].astype('category')
titanic_data["Loc"] =  titanic_data['Embarked'].cat.codes
print(titanic_data.Loc.describe())



# Dependency Analysis

#ANOVA
model = sm.OLS(titanic_data.Age, titanic_data.Pclass, missing="drop").fit() # seems there is a problem on thisone which is the same of ligne 32
model = ols('Age ~ Pclass', data=titanic_data, missing="drop").fit()
anova = sm.stats.anova_lm(model)
print(anova)

mod = ols('Fare ~ Pclass', data=titanic_data, missing="drop").fit()
an = sm.stats.anova_lm(mod)
print(an)

model1 = ols('Fare ~ Sex', data=titanic_data, missing="drop").fit()
anova1 = sm.stats.anova_lm(model1)
print(anova1)


#Chi_2
a=pd.crosstab(index=titanic_data['gender'],columns=titanic_data['Pclass'])
print(a)
b=stats.chi2_contingency(a)
print(b)
#We reject the null of indepedence

a=pd.crosstab(index=titanic_data['Pclass'],columns=titanic_data['Survived'])
print(a)
b=stats.chi2_contingency(a)
print(b)
#We reject the null of indepedence


a=pd.crosstab(index=titanic_data['gender'],columns=titanic_data['Survived'])
print(a)
b=stats.chi2_contingency(a)
print(b)
#We reject the null of indepedence


#Correlation
print(titanic_data.plot.scatter("Age","Fare"))
titanic_l=titanic_data[["Age", "Fare"]].dropna()
corrMatrix = titanic_l.corr()
print(corrMatrix)
print(stats.pearsonr(titanic_l.Age, titanic_l.Fare)) # correlation is bias by the sample size 
#conclusion: we reject Ho
# So, there is a strong correlation
