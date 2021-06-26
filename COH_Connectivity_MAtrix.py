# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 19:11:30 2021

@author: Neptune
"""

# Importing necassary libraries
import numpy as np
import scipy.signal as ss
from nilearn.connectome import ConnectivityMeasure

def Connectivity_Matrix(Epoch_Object):
    conectivity_matrix = []
    # Gettig data from epoch object in array form
    Epoch_Object = Epoch_Object.get_data(picks='all')
    
    # No of epochs, channels and datapoints/epoch
    epochs = Epoch_Object.shape[0]
    channels = Epoch_Object.shape[1]
    data_points = Epoch_Object.shape[2]
    
    # Predefining connectivity matrix
    conectivity_matrix_beta = np.zeros([epochs,channels,channels],dtype=float)
    conectivity_matrix_betaN = np.zeros([epochs,channels,channels],dtype=float)
    conectivity_matrix_theta = np.zeros([epochs,channels,channels],dtype=float)
    conectivity_matrix_thetaN = np.zeros([epochs,channels,channels],dtype=float)
    
    for epoch in range(epochs):
        # Getting datapoints correspond to respective epoch
        data = Epoch_Object[epoch,:,:]
        print(epoch)
        for i in range(channels):
            for k in range(channels):
                # print(i)
                f, Cxy = ss.coherence(data[i,:],data[k,:],fs= 250)
                conectivity_matrix_beta[epoch,i,k] = np.mean(Cxy[np.where((f>=15) & (f<=30))])
                conectivity_matrix_betaN[epoch,i,k] = np.mean(Cxy[np.where((f>=22.5) & (f<=25))])
                conectivity_matrix_theta[epoch,i,k] = np.mean(Cxy[np.where((f>=4) & (f<=8))])
                conectivity_matrix_thetaN[epoch,i,k] = np.mean(Cxy[np.where((f>=4) & (f<=6.5))])
                    
            
        
    conectivity_matrix.append(['Beta',conectivity_matrix_beta])
    conectivity_matrix.append(['Beta_Narrow',conectivity_matrix_betaN])
    conectivity_matrix.append(['Theta',conectivity_matrix_theta])
    conectivity_matrix.append(['Theta_Narrow',conectivity_matrix_thetaN])
                
    return conectivity_matrix