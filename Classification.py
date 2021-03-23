# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:18:44 2021

@author: Muhammad Salman Kabir
@purpose: Within Subject Classification
@regarding: Functional Connectivity Analysis
"""

import sklearn as sk

def Classification(Feature_Table):
    ## ------------------------------------------------------------------------
    # Classification do the SVM classification of time series data based on PLI 
    # features
    # Input -->
    #       Feature_Table: feature table in standard form
    #
    # Output -->
    #       Accuracy
    ##-------------------------------------------------------------------------
    
    # Extracting predictors 
    features = Feature_Table[:,:-1]
    
    # Extracting response
    classes = Feature_Table[:,-1]
    
    # Train Test Data split
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(features, classes, test_size=0.3, random_state =123)
    
    # SVM Classifier init.
    clf = sk.svm.SVC(kernel='poly')
    
    # Fitting
    clf.fit(x_train,y_train)
    
    # Prediction
    y_pred = clf.predict(x_test)
    
    # Accuracy
    return sk.metrics.accuracy_score(y_test, y_pred)*100