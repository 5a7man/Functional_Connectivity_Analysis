# -*- coding: utf-8 -*-
"""
Created on Tue Feb 2 23:11:31 2021

@author: Muhammad Salman Kabir
@purpose: To load epochs_array object
@regarding: Functional Connectivity Analysis
"""

# Importing necassary libraries
import mne

# Loading data
subject_1 = mne.read_epochs('Subject_1-epo.fif', preload=False)

# Reading Data -> (Events,channels,data point per event) 
subject_1_data = subject_1.get_data(picks ='all')

# Plotting Epoch Array
subject_1.plot(picks='all')

