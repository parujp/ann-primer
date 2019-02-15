#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 09:57:35 2019

@author: parujp

Project: Customer-Churn Predictor version 2.0 using a simple ANN

This part focuses on Evaluating the ANN using cross-validation

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


# BUILD, EVALUATE, IMPROVE AND TUNE ANN

# Import NN libraries
import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout

# Evaluating the ANN
# Imports for KFold cross validation 
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

#function that builds  ANN classifier for build function expected as argument in KerasClassifier
def build_classifier():
    # Initialize ANN classifier
    classifier = Sequential()

    # Configure hidden layer with Dropout 
    classifier.add(Dense(units = 6, input_dim = 11, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(units = 6, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dropout(rate=0.1))
    # Add output layer
    classifier.add(Dense(units = 1, kernel_initializer="uniform", activation="sigmoid"))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100 )

#n_jobs = no of CPUs to do the computation. -1 means all CPUs (run llel computations)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)    

mean = accuracies.mean() # 84.162 % 
variance = accuracies.std() # 2% -> high variance?

