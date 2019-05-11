# -*- coding: utf-8 -*-
"""
Proyecto de regresión logística, para determinar si una persona es mayor o no de 50 años
Created on Wed May  1 11:13:30 2019

@author: MS719972
María Teresa Loza Anaya
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb

dataframe = pd.read_csv(r"adult.data-header-renewtypes.csv")
#print(dataframe.head())
#print(dataframe.describe())
#print(dataframe.groupby('age').size())
#dataframe.drop(['finalweigth','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country'],1).hist()
#plt.show()
X=np.array(dataframe.drop(['final'],1))
y=np.array(dataframe['final'])

print(X.shape)

validation_size = 0.30
seed = 7

X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X,y,test_size=validation_size, random_state=seed)

model = linear_model.LogisticRegression()
model.fit(X_train,y_train)

proj_name= "Logistic Regression"
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (proj_name, cv_results.mean(), cv_results.std())
print(msg)

predictions = model.predict(X_validation)
print(model.score(X_validation,y_validation))
print(accuracy_score(y_validation,predictions))

print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation,predictions))

#X_probe = pd.DataFrame({'age':[36],'workclass':[89]})

X_probe = pd.DataFrame({'age':[36],'workclass':[89],'finalweigth':[284582],'education':[39],'education-num':[9],'marital-status':[58],'occupation':[78],'relationship':[100],'race':[12],'sex':[34],'capital-gain':[0],'capital-loss':[0],'hours-per-week':[30],'native-country':[61]})
prednew = model.predict(X_probe)
print(prednew)