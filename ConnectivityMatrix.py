# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:56:40 2021

@author: Muhammad Salman Kabir
@purpose: Connectivity Matrix 
@regarding: Functional Connectivity Analysis
"""

# Importing necassary libraries
import numpy as np
import scipy.signal as ss
from nilearn.connectome import ConnectivityMeasure

def Connectivity_Matrix(Epoch_Object,f_min,f_max,Connectivity):
    ## ------------------------------------------------------------------------
    # Connectivity_Matrix computed connectivity (PLI, PLV, CCF) between time series
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
        if Connectivity == "PLI":
            for i in range(channels):
                for k in range(channels):
                    conectivity_matrix[epoch,i,k] = np.sum(np.sign(phase[i,:]-phase[k,:]))/data_points
        
        if Connectivity == "PLV":
            for i in range(channels):
                for k in range(channels):
                    conectivity_matrix[epoch,i,k] = np.sum(np.exp(1j*(phase[i,:]-phase[k,:])))/data_points
                    
        if Connectivity == "CCF":
            for i in range(channels):
                for k in range(channels):
                    temp = np.corrcoef(data[i,:],data[k,:])
                    conectivity_matrix[epoch,i,k] = temp[0][1]
                    
        if Connectivity == "COV":
            data = data.T
            temp = np.reshape(data,(1,data.shape[0],data.shape[1]))
            connectivity_measure = ConnectivityMeasure(kind='covariance')
            conectivity_matrix[epoch,:,:] = connectivity_measure.fit_transform(temp)
            
        if Connectivity == "PCOR":
            data = data.T
            temp = np.reshape(data,(1,data.shape[0],data.shape[1]))
            connectivity_measure = ConnectivityMeasure(kind='partial correlation')
            conectivity_matrix[epoch,:,:] = connectivity_measure.fit_transform(temp)
        
       
        # if Connectivity == "TAN":
        #     data = data.T
        #     connectivity_measure = ConnectivityMeasure(kind='tangent')
        #     conectivity_matrix = connectivity_measure.fit_transform(data)
            
                
    return conectivity_matrix
                
