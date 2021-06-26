

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:18:44 2021

@author: Muhammad Salman Kabir
@purpose: Cross Validation Classification model for high and low 
@regarding: Functional Connectivity Analysis
"""
import numpy as np
import sklearn as sk
import sklearn.model_selection 
import sklearn.svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def CV_Classification(Feature_Table,K,PCA_Flag,PCA_no):
    ## ------------------------------------------------------------------------
    # Classification do the SVM classification of time series data
    # Input -->
    #       Feature_Table: feature table in standard form
    #
    #       Accuracy
    ##-------------------------------------------------------------------------
    
    # Extracting predictors 
    features = (Feature_Table[:,:-1])
    
    # Extracting response
    classes = Feature_Table[:,-1]
    

    if PCA_Flag==1:
        features = StandardScaler().fit_transform(features)
        features = PCA(PCA_no).fit_transform(features)
        
    # SVM Classifier init.
    clf = sk.svm.SVC(kernel='rbf', random_state=42)
    
    # K fold init
    cv = sk.model_selection.KFold(n_splits=K, random_state=1, shuffle=True)
    
    # 
    scores = sk.model_selection.cross_val_score(clf, features, classes, scoring='accuracy', cv=cv)
    
    
    # score return
    return [round(np.mean(scores)*100,2),round(np.std(scores)*100,2)]