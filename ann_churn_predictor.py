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

