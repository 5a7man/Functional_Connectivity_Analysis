# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 19:16:03 2021

@author: Muhammad Salman Kabir
@purpose: Extracting Connectivity from subbjects
@regarding: Functional Connectivity Analysis
"""
import mne
import numpy as np
from ConnectivityTable import Connectivity_Vector

subjects = 20
connectivity_list = ["PLI","PLV","CCF","PCOR"]

for subject in range(subjects):
    print("Subject"+str(subject+1))
    # Loading subject's data
    filename = "Subject_"+ str(subject+1) +"-epo.fif" 
    sub = mne.read_epochs(filename, preload=False)
    
    # Computing connectivity matrix for beta waves
    Connectivity_Table = Connectivity_Vector(sub, 15, 30, connectivity_list)
    
    # Saving feature matrix
    file = "Connectivity_Table_Subject_"+ str(subject+1) +".npy"
    np.save(file,Connectivity_Table)
    