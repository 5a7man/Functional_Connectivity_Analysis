# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:30:04 2021

@author: Muhammad Salman Kabir
@purpose: 
@regarding: Functional Connectivity Analysis
"""
import mne
import numpy as np
from ConnectivityMatrix import Connectivity_Matrix
from Classification import Classification

## Converting Connectivity Matrix into feature table for each subject and 
## saving them
# subjects =  20

# for subject in range(subjects):
#     # Loading subject's data
#     filename = "Subject_"+ str(subject+1) +"-epo.fif" 
#     sub = mne.read_epochs(filename, preload=False)
    
#     # Computing connectivity matrix for beta waves
#     connectivity_matrix = Connectivity_Matrix(sub,15,30)
    
#     # Forming Feature Table
#     feature_table = np.zeros([connectivity_matrix.shape[0],466])
#     for epoch in range(connectivity_matrix.shape[0]):
#         k = 0
#         for i in range(connectivity_matrix.shape[1]-1):
#             for j in range(connectivity_matrix.shape[2]-i-1):
#                 feature_table[epoch,k] = connectivity_matrix[epoch,i,j+i+1] 
#                 k = k+1
    
#     # Class assignment -> 0: begin, 1: end
#     feature_table[40:,465] = 1
    
#     # Saving feature table
#     file = "Connectivity_Subject_"+ str(subject+1) +".npy"
#     np.save(file,feature_table)
    

## Classification

# Forming Feature Table by combining features of all subjects
# feature_table = np.zeros([1,466])
# subjects = 1
# for subject in range(subjects):
#     filename = 'Connectivity_Subject_'+ str(subject+1) + '.npy'
#     data = np.load(filename)
#     feature_table = np.vstack([feature_table,data])
     
# feature_table = feature_table[1:,:]    

# Accuracy = Classification(feature_table)
# print('Accuracy is:',Accuracy,'%')


### Classification Accuracy Per Subject
filename = 'Connectivity_Subject_'+ str(1) + '.npy'
feature_table = np.load(filename)
Accuracy = Classification(feature_table)
print('Accuracy is:',Accuracy,'%')
