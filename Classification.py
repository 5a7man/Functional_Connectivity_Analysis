# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:18:44 2021

@author: Muhammad Salman Kabir
@purpose: Classification model for high and low 
@regarding: Functional Connectivity Analysis
"""

import sklearn as sk
import sklearn.model_selection
import sklearn.svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def Classification(Feature_Table,PCA_Flag,PCA_no):
    ## ------------------------------------------------------------------------
    # Classification do the SVM classification of time series data
    # Input -->
    #       Feature_Table: feature table in standard form
    #
    # Output -->
    #       Accuracy
    ##-------------------------------------------------------------------------
    
    # Extracting predictors 
    features = (Feature_Table[:,:-1])
    
    # Extracting response
    classes = Feature_Table[:,-1]
    
    
    # Train Test Data split
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(features, classes, test_size=0.3, random_state =123)
    
    if PCA_Flag==1:
        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)
        pca = PCA(PCA_no).fit(x_train)
        x_train =pca.transform(x_train)
        x_test = pca.transform(x_test)
        
    # SVM Classifier init.
    clf = sk.svm.SVC(kernel='rbf')
    
    # Fitting
    clf.fit(x_train,y_train)
    
    # Prediction
    y_pred = clf.predict(x_test)
    
    # Accuracy
    return sk.metrics.accuracy_score(y_test, y_pred)*100