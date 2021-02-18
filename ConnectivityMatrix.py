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


def Connectivity_Matrix(Epoch_Object):
    ## ------------------------------------------------------------------------
    # Connectivity_Matrix computed connectivity (PLI based) between time series
    # Input -->
    #       Epoch_Object: in form (epochs,channels,data_points)
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
    
    
    for epoch in range(epochs):
        # Getting datapoints correspond to respective epoch
        data = Epoch_Object[epoch,:,:]
        
        # Hilbert transform
        data_hilbert = np.imag(ss.hilbert(data))
        
        # Phase computation
        phase = np.arctan(data_hilbert/data)
        
        # Connectivity Matrix Computation
        for i in range(channels):
            for j in range(channels):
                conectivity_matrix[epoch,i,j] = np.sum(np.sign(phase[i,:]-phase[j,:]))/data_points
                
    return conectivity_matrix
                
  
## Testing 
# Loading epoch object
subject_1 = mne.read_epochs('Subject_1-epo.fif', preload=False)

# Finding connectivity matrix (PLI based) 
conectivity_matrix = Connectivity_Matrix(subject_1)



    
    
    
    
    
    
    
    
    
    