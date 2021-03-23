# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:56:40 2021

@author: Muhammad Salman Kabir
@purpose: Connectivity Matrix (PLI)
@regarding: Functional Connectivity Analysis
"""

# Importing necassary libraries
import mne
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

def Connectivity_Matrix(Epoch_Object,f_min,f_max):
    ## ------------------------------------------------------------------------
    # Connectivity_Matrix computed connectivity (PLI based) between time series
    # Input -->
    #       Epoch_Object: in form (epochs,channels,data_points)
    #       f_min: lowest frequency of band in hertz
    #       f_max: maximum frequency of band in hertz
    #
    # Output -->
    #       connectivity_matrix: in form (epochs,channels,channels)
    ##-------------------------------------------------------------------------    
    
    # Gettig data from epoch object in array form
    Epoch_Object = Epoch_Object.get_data(picks='all')
    
    # No of epochs, channels and datapoints/epoch
    epochs = Epoch_Object.shape[0]
    channels = Epoch_Object.shape[1]
    data_points = Epoch_Object.shape[2]
    
    # Predefining connectivity matrix
    conectivity_matrix = np.zeros([epochs,channels,channels],dtype=float)
    
    # Designing Filter
    sos = ss.butter(N=10,Wn=[f_min,f_max],btype='bandpass',analog=False,output='sos',fs=250)
    
    
    for epoch in range(epochs):
        # Getting datapoints correspond to respective epoch
        data = Epoch_Object[epoch,:,:]
        
        # Filteration
        data = ss.sosfilt(sos, data)
        
        # Hilbert transform
        data_hilbert = np.imag(ss.hilbert(data))
        
        # Phase computation
        phase = np.arctan(data_hilbert/data)
        
        # Connectivity Matrix Computation
        for i in range(channels):
            for j in range(channels):
                conectivity_matrix[epoch,i,j] = np.sum(np.sign(phase[i,:]-phase[j,:]))/data_points
                
    return conectivity_matrix
                
  