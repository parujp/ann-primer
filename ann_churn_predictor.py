#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 00:29:21 2019

@author: parujp

Project: Customer-Churn Predictor using a simple ANN

"""
# DATA PREPROCESSING

# Import ds libraries

import numpy as np
import pandas as pd

# Get the data

dataset = pd.read_csv("Churn_data.csv")
#dataset.columns

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values    
                
# Encode categorical data - Geography , Gender

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Geography
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
#Gender
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
# since we have 3 countries
#dataset['Geography'].value_counts()
onehotencoder = OneHotEncoder(categorical_features = [1] ) # index 1
X = onehotencoder.fit_transform(X).toarray()
# remove the left most vestigial dummy variable
X = X[:,1:]

# Train-test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=12)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# BUILDING THE ANN

# Import NN libraries
import keras
from keras.models import Sequential 
from keras.layers import Dense

# The ANN for churn prediction is a classifier

# Initialize ANN classifier
classifier = Sequential()

# Configure the first hidden layer

# Rule of thumb : no of nodes in hidden layer = average og no of i/p and no of o/ps = (11 + 1 )/2
# units = no of nodes
# input_dim = no of independent variables or i/p nodes
# kernel_initializer = 'uniform' : Initialize weights using uniform distribution
# activation = 'relu' => use rectifier function for activation

classifier.add(Dense(units = 6, input_dim = 11, kernel_initializer='uniform', activation='relu'))

# Configure second hidden layer ( Unnecessary)

classifier.add(Dense(units = 6, activation="relu", kernel_initializer="uniform"))

# Add output layer

classifier.add(Dense(units = 1, kernel_initializer="uniform", activation="sigmoid"))

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting ANN to training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
 # ACCURACY : 83.91%
 
 # MAKING PREDICTIONS AND EVALUATING THE MODEL
 
 # Predict test set results
 y_pred = classifier.predict(X_test) # is probabilities of churn
 y_pred = (y_pred > 0.50)
 
 # Make confusion matrix
 from sklearn.metrics import confusion_matrix
 cm = confusion_matrix(y_test, y_pred)
 # Accuracy = (1530+150)/2000 = 84% ...yay!
 
# Predict true/false for a new customer
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

new_customer = sc.transform(np.array([[0.0,0,600,1,40,3,6000,2,1,1,50000]]))
new_prediction = classifier.predict(new_customer)
new_prediction = (new_prediction > 0.50) 
# false 


