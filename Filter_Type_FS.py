# -*- coding: utf-8 -*-
"""
Created on Sat May 22 00:06:15 2021

@author: Muhammad Salman Kabir
@purpose: Feature_Engineering (Filter Type) 
@regarding: Functional Connectivity Analysis
"""

# change filename (line 18) and score_func (line 28)
import numpy as np
from Classification import Classification
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import pandas as pd



filename = 'Feature_Table_CCF.npy'
feature_table = np.load(filename)


K = [20,40,60,80,100,120]
Accuracy = np.zeros((len(K)))
for i in range(len(K)):
    d = pd.DataFrame(feature_table)
    X = d.iloc[:,:-1]
    y=d.iloc[:,-1]
    bestfeatures = SelectKBest(score_func=f_classif, k=K[i])
    fit = bestfeatures.fit(abs(X),y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  
    print(featureScores.nlargest(10,'Score'))  
    
    imp = featureScores.nlargest(K[i],'Score').Specs
    imp = imp.to_numpy()
    imp = np.hstack([imp,465])
    data = feature_table[:,imp]
    Accuracy[i] = Classification(data,0,0)

