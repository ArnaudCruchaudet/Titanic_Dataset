#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:47:12 2024

@author: arnaudcruchaudet
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import graphviz 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics 

titanic_data = pd.read_csv("titanic.csv", header = 0, sep = ",")
titanic=titanic_data
titanic_data["gender"] = np.where(titanic_data["Sex"].str.contains("female"), 1, 0)

print (titanic_data.head())

titanic_B=titanic_data.dropna(subset=["Age"])

X=titanic_B[["Pclass", "Fare", "Age", "gender", "SibSp", "Parch"]]
Y=titanic_B[["Survived"]]



########################################### DECISION TREE ALGO ##################################""



model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(X,Y)
print(tree.export_text(model))
tree.plot_tree(model)
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(X,Y)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#Pre-Prunning

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
model = tree.DecisionTreeClassifier(max_leaf_nodes=5, random_state = 0)
model.fit(X,Y)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print(tree.export_text(model))
tree.plot_tree(model)
plt.show()


dot_data = tree.export_graphviz(model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph




#modifying Y into string in order to represent the tree
Y[["Survived"]] = Y[["Survived"]].replace({'Survived': {1: 'yes', 0: 'no'}})
y_class_names=list(Y['Survived'].unique())
dot_data = tree.export_graphviz(model, out_file=None, 
                  feature_names=X.columns, class_names=y_class_names, 
                   filled=True)  
graph = graphviz.Source(dot_data) 
graph


#other way to produce the Tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model, 
                   feature_names=X.columns,  
                   class_names=y_class_names,
                   filled=True)
print(fig)




X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
model = tree.DecisionTreeClassifier(max_depth=5, random_state = 0)
model.fit(X,Y)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print(tree.export_text(model))
tree.plot_tree(model)
plt.show()


#Post pruning

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
model1 = tree.DecisionTreeClassifier(random_state = 0)
path1 = model1.cost_complexity_pruning_path(X_train, y_train)
path1
ccp_alphas, impurities = path1.ccp_alphas, path1.impurities
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, impurities)
plt.xlabel("effective alpha")
plt.ylabel("total impurity of leaves")

#As we can see higher alpha is associated with higher impurity


#Now we train a decision tree using the effective alphas. 
clfs = []

for ccp_alpha in ccp_alphas:
    model1 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    model1.fit(X_train, y_train)
    clfs.append(model1)

tree_depths = [model1.tree_.max_depth for model1 in clfs]
plt.figure(figsize=(10,  6))
plt.plot(ccp_alphas[:-1], tree_depths[:-1])
plt.xlabel("effective alpha")
plt.ylabel("total depth")

#This last graph show how depth decreases as alpha increases
# we remove the last element in clfs and ccp_alphas, because it is the trivial tree with only one node

acc_scores = [metrics.accuracy_score(y_test, model1.predict(X_test)) for model1 in clfs]
tree_depths = [model1.tree_.max_depth for model1 in clfs]
plt.figure(figsize=(10, 10))
plt.grid()
plt.plot(ccp_alphas[:-1], acc_scores[:-1])
plt.xlabel("effective alpha")
plt.ylabel("Accuracy scores")


#This last graph plots the performance of the Decision Tree according to effective alpha

###To check appropriate value for alpha***
acc_scores = [metrics.accuracy_score(y_test, model1.predict(X_test)) for model1 in clfs]
tree_depths = [model1.tree_.max_depth for model1 in clfs]
plt.figure(figsize=(15, 6))
plt.grid()
plt.plot(ccp_alphas[:-1], acc_scores[:-1])
plt.xlabel("effective alpha")
plt.ylabel("Accuracy scores")
plt.xticks(np.arange(0, 0.02, 0.002))

###Final Decision Tree***
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
model1 = tree.DecisionTreeClassifier(random_state = 0, ccp_alpha=0.007)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
tree.plot_tree(model1)
plt.show()






########################################### RANDOM FOREST ALGO ##################################""




#Random forest classifier
# Averaging remove part of the noise 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
model1 = RandomForestClassifier(n_estimators = 100) 
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
model1 = RandomForestClassifier(n_estimators = 100, max_leaf_nodes=5) 
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
model1 = RandomForestClassifier(n_estimators = 1000, ccp_alpha=0.002) 
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# Importance of different features 
feature_imp = pd.Series(model1.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp


#Random forest regressor
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
XX = housing.data[:,:]  
y = housing.target
XX2 = StandardScaler().fit_transform(XX)



X_train, X_test, y_train, y_test = train_test_split(XX2, y, test_size = 0.20)
model1 = RandomForestRegressor(n_estimators = 100) 
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(metrics.r2_score(y_test, y_pred))




X_train, X_test, y_train, y_test = train_test_split(XX2, y, test_size = 0.20)
model1 = RandomForestRegressor(n_estimators = 100, max_leaf_nodes=5)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(metrics.r2_score(y_test, y_pred))


X_train, X_test, y_train, y_test = train_test_split(XX2, y, test_size = 0.20)
model1 = RandomForestRegressor(n_estimators = 500, max_depth=7 ) 
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(metrics.r2_score(y_test, y_pred))


feature_imp = pd.Series(model1.feature_importances_,index=housing.feature_names).sort_values(ascending=False)
print(feature_imp)

####Implementing Cost-Complexity Pruning

X_train, X_test, y_train, y_test = train_test_split(XX2, y, test_size = 0.20)
model1 = tree.DecisionTreeRegressor(random_state = 0)
path1 = model1.cost_complexity_pruning_path(X_train, y_train)
path1
ccp_alphas, impurities = path1.ccp_alphas, path1.impurities
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, impurities)
plt.xlabel("effective alpha")
plt.ylabel("total impurity of leaves")

#As we can see higher alpha is associated with higher impurity


#Now we train a decision tree using the effective alphas. 
clfs = []

for ccp_alpha in ccp_alphas:
    model1 = tree.DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
    model1.fit(X_train, y_train)
    clfs.append(model1)

tree_depths = [model1.tree_.max_depth for model1 in clfs]
plt.figure(figsize=(10,  6))
plt.plot(ccp_alphas[:-1], tree_depths[:-1])
plt.xlabel("effective alpha")
plt.ylabel("total depth")


#This last graph show how depth decreases as alpha increases
# we remove the last element in clfs and ccp_alphas, because it is the trivial tree with only one node

acc_scores = [metrics.accuracy_score(y_test, model1.predict(X_test)) for model1 in clfs]
tree_depths = [model1.tree_.max_depth for model1 in clfs]
plt.figure(figsize=(10, 10))
plt.grid()
plt.plot(ccp_alphas[:-1], acc_scores[:-1])
plt.xlabel("effective alpha")
plt.ylabel("Accuracy scores")


#This last graph plots the performance of the Decision Tree according to effective alpha

###To check appropriate value for alpha***
acc_scores = [metrics.accuracy_score(y_test, model1.predict(X_test)) for model1 in clfs]
tree_depths = [model1.tree_.max_depth for model1 in clfs]
plt.figure(figsize=(15, 6))
plt.grid()
plt.plot(ccp_alphas[:-1], acc_scores[:-1])
plt.xlabel("effective alpha")
plt.ylabel("Accuracy scores")
plt.xticks(np.arange(0, 0.02, 0.002))