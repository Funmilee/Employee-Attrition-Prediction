# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:08:01 2020

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing dataset
dataset = pd.read_csv("Attrition_Data.csv")
# I dropped Left column because it is the dependent dataset and department is not needed
X = dataset.drop(["Left", "Department"], axis=1)
Y = dataset.iloc[:, 6].values

#Encoding the Categorical data(Salary Column)
d = {'Low': 0,'Medium': 1,'High': 2}
X["salary"] = X["salary"].map(d)
#dataset["salary"]= dataset["salary"].map(d)
# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=0)

#Exploratory Data Analysis (trying to see the correlation between independent variable and dependent variable 
#Visual plot of satisfaction level against turnover("Left" column) 
 satisfaction_level_plot = sns.boxplot(x="Left", y= "satisfaction_level", data= dataset)
plt.show()
figure = satisfaction_level_plot.get_figure() 
figure.savefig('satisfaction_level_plot.png')

#Visual plot of last evaluation against turnover("Left" column) 
last_evaluation_plot = sns.boxplot(x="Left", y= "last_evaluation", data= dataset)
plt.show()
figure = last_evaluation_plot.get_figure()
figure.savefig('last_evaluation_plot.png')

#Visual plot number_project against turnover("Left" column) 
number_project_plot = sns.boxplot(x="Left", y= "number_project", data= dataset)
plt.show()
figure = number_project_plot.get_figure()
figure.savefig('number_project_plot.png')

#Visual plot average_montly_hours against turnover("Left" column)
average_montly_hours_plot = sns.boxplot(x="Left", y= "average_montly_hours", data= dataset)
plt.show()
figure = average_montly_hours_plot.get_figure()
figure.savefig('average_montly_hours_plot.png')

#Visual plot time_spend_company against turnover("Left" column)
time_spend_company_plot = sns.boxplot(x="Left", y= "time_spend_company", data= dataset)
plt.show()
figure = time_spend_company_plot.get_figure()
figure.savefig('time_spend_company_plot.png')

#Visual plot Work_accident against turnover("Left" column)
Work_accident_plot = sns.boxplot(x="Left", y= "Work_accident", data= dataset)
plt.show()
figure = Work_accident_plot.get_figure()
figure.savefig('Work_accident_plot.png')

#Visual plot promotion_last_5years against turnover("Left" column)
promotion_last_5years_plot = sns.boxplot(x="Left", y= "promotion_last_5years", data= dataset)
plt.show()
figure = promotion_last_5years_plot.get_figure()
figure.savefig('promotion_last_5years_plot.png')

#Visual plot salary against turnover("Left" column)
salary_plot = sns.boxplot(x="Left", y= "salary", data= dataset)
plt.show()
figure = salary_plot.get_figure()
figure.savefig('salary_plot.png')
# Fitting Decision Tree model to Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

# Predicting decision Tree Test set results
Y_pred_Decision_Tree = classifier.predict(X_test)

# Confusion Matrix for Decision Tree Model
from sklearn.metrics import confusion_matrix
cm_Decision_Tree = confusion_matrix(Y_test, Y_pred_Decision_Tree)
#Accuracy score calculation for Decision Tree Model
from sklearn.metrics import accuracy_score
acc_decision_tree = accuracy_score(Y_test,Y_pred_Decision_Tree)
print(acc_decision_tree)

# Fitting Naive Bayes Algorithm to Training set
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_naive_bayes = sc.fit_transform(X_train)
X_test_naive_bayes = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train_naive_bayes, Y_train)

# Predicting naive_bayes Test set results
Y_pred_naive_bayes = classifier.predict(X_test_naive_bayes)

# Confusion Matrix for naive_bayes_model
from sklearn.metrics import confusion_matrix
cm_naive_bayes = confusion_matrix(Y_test, Y_pred_naive_bayes)
#Accuracy score calculation for naive_bayes_model
from sklearn.metrics import accuracy_score
acc_naives_bayes = accuracy_score(Y_test,Y_pred_naive_bayes)
print(acc_naives_bayes)

# Fitting Random Forest Classification to Training set
from sklearn.ensemble import RandomForestClassifier
classifier  = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

# Predicting Test set results
Y_pred_RandomForest = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RandomForest = confusion_matrix(Y_test, Y_pred_RandomForest)
from sklearn.metrics import accuracy_score
acc_random_forest = accuracy_score(Y_test,Y_pred_RandomForest)
print(acc_random_forest)

