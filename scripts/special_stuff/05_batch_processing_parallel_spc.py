#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:19:58 2023

@author: krohn
"""


'''
Essentially the same as 04_batch_processing_parallel.py, except used on some 
.spc files. Currently only doing auto-correlations, have not yet decided how to 
implement cross-correlation of .spc data.

'''


# For I/O
import os
import sys
import glob

# For localizing FCS_Fixer
repo_dir = os.path.abspath('..')

# For data processing
sys.path.insert(0, repo_dir)
from functions import FCS_Fixer
import numpy as np

# misc
import warnings



#%% Input data
# You can extend dir_names with as many source directories as you want
dir_names=[]
dir_names.extend(['some_base_dir/siRNA_in_buffer'])


#%% Settings

# Basic settings for correlation
tau_min = 1E-6
tau_max = 1.0
sampling = 6
cross_corr_symm = True
correlation_method = 'default'
default_uncertainty_method = 'Bootstrap'

# We run auto-correlation within all three PIE gates
list_of_channel_pairs = [(FCS_Fixer.FCS_Fixer.build_channels_spec(channels_indices = 0,
                                                                  micro_time_gates = np.array([0., 1/3])),
                          FCS_Fixer.FCS_Fixer.build_channels_spec(channels_indices = 0,
                                                                  micro_time_gates = np.array([0., 1/3]))),
                         (FCS_Fixer.FCS_Fixer.build_channels_spec(channels_indices = 0,
                                                                 micro_time_gates = np.array([1/3, 2/3])),
                          FCS_Fixer.FCS_Fixer.build_channels_spec(channels_indices = 0,
                                                                  micro_time_gates = np.array([1/3, 2/3]))),
                         (FCS_Fixer.FCS_Fixer.build_channels_spec(channels_indices = 0,
                                                                 micro_time_gates = np.array([2/3, 1.])),
                          FCS_Fixer.FCS_Fixer.build_channels_spec(channels_indices = 0,
                                                                  micro_time_gates = np.array([2/3, 1.]))),
                         ] # Empty list = auto-detect and use all options

# How many parallel processes?
process_count = os.cpu_count() // 2

# Which filters to use?
use_calibrated_AP_subtraction = False
afterpulsing_params_path = os.path.join(repo_dir, 'functions/ap_params_2ch_setup.csv')
use_burst_removal = False
use_bleaching_correction = False
use_mse_filter = False
use_flcs_bg_subtraction = True


# Where to collect results?
out_dir = ''

#%% Go through diretories and find all .ptu files, creating effectively pairs of directory, file for each of them
_file_names=[]
_dir_names = []
for dir_name in dir_names:
    for file_name in glob.glob(dir_name+'/*ch*.spc'):
        _, name = os.path.split(file_name)

        _file_names.extend([name])
        _dir_names.extend([dir_name])


# #%% Iterate over data
in_paths=[os.path.join(_dir_names[i],file_name) for i, file_name in enumerate(_file_names)]


scheduler = FCS_Fixer.Parallel_scheduler(in_paths,
                                          tau_min = tau_min,
                                          tau_max = tau_max,  
                                          sampling = sampling,
                                          correlation_method = correlation_method,
                                          cross_corr_symm = cross_corr_symm,
                                          use_calibrated_AP_subtraction = use_calibrated_AP_subtraction,
                                          afterpulsing_params_path = afterpulsing_params_path,
                                          list_of_channel_pairs = list_of_channel_pairs,
                                          use_burst_removal = use_burst_removal,
                                          use_drift_correction = use_bleaching_correction,
                                          use_mse_filter = use_mse_filter,
                                          use_flcs_bg_corr = use_flcs_bg_subtraction,
                                          default_uncertainty_method = default_uncertainty_method,
                                          out_dir = out_dir
                                          )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scheduler.run_parallel_processing(process_count)

print('Job done.')