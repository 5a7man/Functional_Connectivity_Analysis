# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:30:04 2021

@author: Muhammad Salman Kabir
@purpose: 
@regarding: Functional Connectivity Analysis
"""


import numpy as np
from Classification import Classification



### Classification Accuracy

filename = 'Feature_Table_PLI.npy'
feature_table = np.load(filename)
Accuracy = Classification(feature_table)
print('Accuracy is:',Accuracy,'%')

### Classification Accuracy
# Accuracy = np.zeros((20))
# for subject in range(20):
#     filename = 'Connectivity_Table_Subject_'+ str(subject+1) + '.npy'
#     feature_table = np.load(filename,allow_pickle=True)[3,1]
#     Accuracy[subject] = Classification(feature_table)
    