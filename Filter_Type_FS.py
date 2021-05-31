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
from CV_Classification import CV_Classification
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import pandas as pd



# filename = 'Feature_Table_CCF.npy'
# feature_table = np.load(filename)


# K = [20,40,60,80,100,120]

features_names_beta = np.load("Feature_Name_Beta.npy")
features_names_theta = np.load("Feature_Name_Theta.npy")
Feature_Names = np.hstack([features_names_beta,features_names_theta,"Class"]) 
K = 0 # 0: PLI, 1:PLV, 2: CCF, 3: PCOR
Accuracy = np.zeros((20,2))
for subject in range(1):# set range according to subject no
    filename = 'Feature_Table_Subject_Beta_'+ str(subject+1) + '.npy'
    temp1 = np.load(filename,allow_pickle=True)[K,1]
    temp1 = temp1[:,:-1]
    filename = 'Feature_Table_Subject_Theta_'+ str(subject+1) + '.npy'
    temp2 = np.load(filename,allow_pickle=True)[K,1]
    feature_table = np.hstack([temp1,temp2])
    d = pd.DataFrame(feature_table,columns=Feature_Names)
    X = d.iloc[:,:-1]
    y=d.iloc[:,-1]
    bestfeatures = SelectKBest(score_func=chi2, k=140)
    fit = bestfeatures.fit(abs(X),y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  
    top_features=featureScores.nlargest(20,'Score')
    atop_features=top_features.Specs.tolist()
    print(featureScores.nlargest(20,'Score'))  
    
#     imp = featureScores.nlargest(140,'Score').Specs
#     imp = imp.to_numpy()
#     imp = np.hstack([imp,930])
#     data = feature_table[:,imp]
#     Accuracy[subject,:] = CV_Classification(data,10,0,0)

