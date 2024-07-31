#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:19:58 2023

@author: krohn
"""


'''
This script finally is similar to 03_batch_processing.py, but accelerates 
processing through parallel computing. 

For this we use another smaller class within FCS_Fixer, called Parallel_scheduler.
Parallel_scheduler accepts essentially the same "global" parameters as FCS_Fixer 
itself, with the addition of in_paths, which is a list of paths to the files to process,
and some added technical parameters.

Parallel_scheduler.run_parallel_processing() then does essentially what 03_batch_processing.py
does. Currently you have no access to the process parameters except those that 
are listed in this script already, though. In other words, all filters are run 
with default settings, and no micro time gating is applied.

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

# misc
import warnings

#%% Input data
# You can extend dir_names with as many source directories as you want
dir_names=[]
dir_names.extend(['../test_data/2ch_setup'])


#%% Settings

# Basic settings for correlation
tau_min = 1E-6
tau_max = 1.0
sampling = 6
cross_corr_symm = True
correlation_method = 'default'
default_uncertainty_method = 'Wohland'
list_of_channel_pairs = [] # Empty list = auto-detect and use all options

# How many parallel processes?
process_count = os.cpu_count()

# Which filters to use?
use_calibrated_AP_subtraction = True
afterpulsing_params_path = os.path.join(repo_dir, 'functions/ap_params_2ch_setup.csv')
use_burst_removal = True
use_bleaching_correction = True
use_mse_filter = True
use_flcs_bg_subtraction = True


# Where to collect results?
out_dir = ''

#%% Go through diretories and find all .ptu files, creating effectively pairs of directory, file for each of them
_file_names=[]
_dir_names = []
for dir_name in dir_names:
    for file_name in glob.glob(dir_name+'/*.ptu'):
        _, name = os.path.split(file_name)

        _file_names.extend([name])
        _dir_names.extend([dir_name])


#%% Iterate over data
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