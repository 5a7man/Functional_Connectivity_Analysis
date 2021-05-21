# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:11:17 2021

@author: Muhammad Salman Kabir
@purpose: Forming Feature Vector from Connectivity Matrix
@regarding: Functional Connectivity Analysis
"""
import mne
import numpy as np
from ConnectivityMatrix import Connectivity_Matrix

def Connectivity_Vector(subject,fmin,fmax,connectivity_list):
    Connectivity_Table = []
    for connectivity in range(len(connectivity_list)):
        
        # Computing connectivity matrix for beta waves
        connectivity_matrix = Connectivity_Matrix(subject,15,30,connectivity_list[connectivity])
        
        # Forming Feature Table
        connectivity_vector = np.zeros([connectivity_matrix.shape[0],466])
        for epoch in range(connectivity_matrix.shape[0]):
            k = 0
            for i in range(connectivity_matrix.shape[1]-1):
                for j in range(connectivity_matrix.shape[2]-i-1):
                    connectivity_vector[epoch,k] = connectivity_matrix[epoch,i,j+i+1] 
                    k = k+1
        
        # Class assignment -> 0: begin, 1: end
        connectivity_vector[40:,465] = 1
        Connectivity_Table.append([connectivity_list[connectivity],connectivity_vector])
        
    # Returnning table
    return Connectivity_Table