# -*- coding: utf-8 -*-
"""
Created on Sat May 22 00:06:15 2021

@author: Neptune
"""

import numpy as np
from Classification import Classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, SelectPercentile
import pandas as pd



filename = 'Feature_Table_PLV.npy'
feature_table = np.load(filename)
# imp =
# data = feature_table[:,(1,4)]



d = pd.DataFrame(feature_table)
X = d.iloc[:,:-1]
y=d.iloc[:,-1]

bestfeatures = SelectKBest(score_func=chi2, k=20)
fit = bestfeatures.fit(abs(X),y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  
print(featureScores.nlargest(20,'Score'))  

imp = featureScores.nlargest(20,'Score').Specs
imp = imp.to_numpy()
imp = np.hstack([imp,465])
data = feature_table[:,imp]
Accuracy = Classification(data)
print('Accuracy is:',Accuracy,'%')
