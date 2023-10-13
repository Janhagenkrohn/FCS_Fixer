#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:19:58 2023

@author: krohn
"""


'''
This script basically does the same thing as 02_standard_pipeline.ipynb, but is written
to auto-detect all .ptu files in one or multiple source directories, and automatically 
process all of them for all channel combination present in the respective file.

Currently all FCS_Fixer methods are used with default settings. Compare the
function calls to those in the Jupyter Notebook to figure out which keyword arguments
you can use to adapt the script to your data, if the default parameters do not work well.
'''


# For I/O
import os
import sys
import glob

# For localizing FCS_Fixer
repo_dir = os.path.abspath('..')

# For data processing
sys.path.insert(0, repo_dir)
import tttrlib
from functions import FCS_Fixer
import numpy as np

# Other
import datetime # For creating time stamps
import traceback # for raising errors without interrupting code

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

# For log file 
script_name = '03_batch_processing.py'

# Which filters to use?
use_calibrated_AP_subtraction = True
afterpulsing_params_path = os.path.join(repo_dir, 'functions/ap_params_2ch_setup.csv')
use_burst_removal = True
use_bleaching_correction = True
use_mse_filter = True
use_flcs_bg_subtraction = True

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
failed_path=[]

for i_file, in_path in enumerate(in_paths):

    try:
        print('Processing' + in_path + '...')
        photon_data = tttrlib.TTTR(in_path,'PTU')
    
        out_name_common = os.path.splitext(_file_names[i_file])[0]
        out_path = os.path.join(_dir_names[i_file], datetime.datetime.now().strftime("%Y%m%d_%H%M") +'_'+ out_name_common)
    
        fixer = FCS_Fixer.FCS_Fixer(photon_data = photon_data, 
                                                    out_path = out_path,
                                                    tau_min = tau_min,
                                                    tau_max = tau_max,
                                                    sampling = sampling,
                                                    cross_corr_symm = cross_corr_symm,
                                                    correlation_method = 'default',
                                                    subtract_afterpulsing = use_calibrated_AP_subtraction,
                                                    afterpulsing_params_path = afterpulsing_params_path,
                                                    write_results = True,
                                                    include_header = False,
                                                    write_log = True)
        
        
        # Auto-detect all channels in the file and enumerate all combinations to correlate
        list_of_channel_pairs = fixer.get_channel_combinations(min_photons = 1000)
        
        
        # Perform all correlations
        for channels_spec_1, channels_spec_2 in list_of_channel_pairs:
            
            try: 
                is_cross_corr = channels_spec_1 != channels_spec_2
                
                # First correlation: No filters
                _ = fixer.get_correlation_uncertainty(channels_spec_1,
                                                        channels_spec_2, 
                                                        use_drift_correction = False,
                                                        use_flcs_bg_corr = False,
                                                        use_burst_removal = False,
                                                        use_mse_filter = False,
                                                        calling_function = script_name)
                
                # First filter: Burst removal
                if use_burst_removal:
                    
                    # Auto-tune time trace bin width
                    time_trace_sampling = fixer.get_trace_time_scale(channels_spec_1,
                                                                        calling_function = script_name)
                    
                    if is_cross_corr:
                        # If we have two-channel data, let's use a geometric mean of 
                        # time_trace_sampling suggestions for two the two channels as compromise
                        time_trace_sampling = np.sqrt(time_trace_sampling * fixer.get_trace_time_scale(channels_spec_2,
                                                                                                         calling_function = script_name))
                        
                    
    
                    # Get time traces
                    
                    if is_cross_corr:
                        # Two distinct channels
                        time_trace_counts_1, time_trace_t = fixer.get_time_trace(channels_spec_1,
                                                                                    time_trace_sampling,
                                                                                    calling_function = script_name)
                        time_trace_counts_2, _ = fixer.get_time_trace(channels_spec_2,
                                                                                    time_trace_sampling,
                                                                                    calling_function = script_name)
                        
                        # Concatenate for further processing
                        for_reshape = (time_trace_counts_1.shape[0], 1)
                        time_traces = np.concatenate((time_trace_counts_1.reshape(for_reshape), time_trace_counts_2.reshape(for_reshape)), axis = 1)
                        
                    else:
                        # Single channel
                        time_trace_counts, time_trace_t = fixer.get_time_trace(channels_spec_1,
                                                                                    time_trace_sampling,
                                                                                    calling_function = script_name)
                        
                        # We crete a dummy second dimension, although this is not even strictly required
                        time_traces = time_trace_counts.reshape((time_trace_counts.shape[0], 1))
                        
                    # Run actual burst removal
                    _ = fixer.run_burst_removal(time_traces, 
                                                  time_trace_sampling,
                                                  calling_function = script_name)
                
                    # Correlate with filters up to this point applied
                    _ = fixer.get_correlation_uncertainty(channels_spec_1,
                                                            channels_spec_2, 
                                                            use_drift_correction = False,
                                                            use_flcs_bg_corr = False,
                                                            use_burst_removal = use_burst_removal,
                                                            use_mse_filter = False,
                                                            calling_function = script_name)
                # END of burst removal block
                
                # Second filter: bleaching/drift correction
                if use_bleaching_correction:
                    
                    # Get time trace
                    time_trace_sampling = fixer.get_trace_time_scale(channels_spec_1,
                                                                       use_burst_removal = use_burst_removal,
                                                                       calling_function = script_name)
                    time_trace_counts, time_trace_t = fixer.get_time_trace(channels_spec_1,
                                                                            time_trace_sampling,
                                                                            use_burst_removal = use_burst_removal,
                                                                            calling_function = script_name)
                    
                    # Run drift/bleaching correction
                    _ = fixer.polynomial_undrifting_rss(time_trace_counts, 
                                                          time_trace_t, 
                                                          channels_spec_1,
                                                          use_burst_removal = use_burst_removal,
                                                          calling_function = script_name)
                    
                    if is_cross_corr:
                        # Second channel
                        
                        # Get time trace
                        time_trace_sampling = fixer.get_trace_time_scale(channels_spec_2,
                                                                           use_burst_removal = use_burst_removal,
                                                                           calling_function = script_name)
                        time_trace_counts, time_trace_t = fixer.get_time_trace(channels_spec_2,
                                                                                time_trace_sampling,
                                                                                use_burst_removal = use_burst_removal,
                                                                                calling_function = script_name)
                        
                        # Run drift/bleaching correction
                        _ = fixer.polynomial_undrifting_rss(time_trace_counts, 
                                                              time_trace_t, 
                                                              channels_spec_2,
                                                              use_burst_removal = use_burst_removal,
                                                              calling_function = script_name)

                    # Correlate with filters up to this point applied
                    _ = fixer.get_correlation_uncertainty(channels_spec_1,
                                                            channels_spec_2, 
                                                            use_drift_correction = use_bleaching_correction,
                                                            use_flcs_bg_corr = False,
                                                            use_burst_removal = use_burst_removal,
                                                            use_mse_filter = False,
                                                            calling_function = script_name)

                # END of bleaching/drift correction block
                
                # Third filter: Removal of anomalous segments based on mse between correlation functions
                if use_mse_filter:
                    
                    # Running this filter is a single function call, whether you have one or two channels
                    _ = fixer.run_mse_filter(channels_spec_1, 
                                                           channels_spec_2,
                                                           use_drift_correction = use_bleaching_correction,
                                                           use_burst_removal = use_burst_removal,
                                                           calling_function = script_name)
                    
                    # Correlate with filters up to this point applied
                    _ = fixer.get_correlation_uncertainty(channels_spec_1,
                                                            channels_spec_2, 
                                                            use_drift_correction = use_bleaching_correction,
                                                            use_flcs_bg_corr = False,
                                                            use_burst_removal = use_burst_removal,
                                                            use_mse_filter = use_mse_filter,
                                                            calling_function = script_name)
                    
                # END of anomalous segment removal block
                
                # Fourth filter: FLCS background subtraction
                if use_flcs_bg_subtraction:
                    
                    # Get TCSPC histogram
                    tcspc_x, tcspc_y = fixer.get_tcspc_histogram(channels_spec_1,
                                                                   use_drift_correction = use_bleaching_correction,
                                                                   use_burst_removal = use_burst_removal,
                                                                   use_mse_filter = use_mse_filter,
                                                                   calling_function = script_name)
                    
                    # Find suitable range for tail fitting, and perform tail fit
                    peak_position = np.argmax(tcspc_y)
                    fit_start = np.uint64(peak_position + np.ceil(2E-9 / fixer.micro_time_resolution))
                    flat_background, _ = fixer.get_background_tail_fit(channels_spec_1, 
                                                                         peak_position, 
                                                                         fit_start,
                                                                         use_drift_correction = use_bleaching_correction,
                                                                         use_burst_removal = use_burst_removal,
                                                                         use_mse_filter = use_mse_filter,
                                                                         calling_function = script_name)
                    
                    # Get FLCS weights
                    _ = fixer.get_flcs_background_filter(tcspc_x, 
                                                           tcspc_y, 
                                                           flat_background, 
                                                           channels_spec_1,
                                                           calling_function = script_name)
                    
                    if is_cross_corr:
                        # Second channel
                        
                        # Get TCSPC histogram
                        tcspc_x, tcspc_y = fixer.get_tcspc_histogram(channels_spec_2,
                                                                       use_drift_correction = use_bleaching_correction,
                                                                       use_burst_removal = use_burst_removal,
                                                                       use_mse_filter = use_mse_filter,
                                                                       calling_function = script_name)
                        
                        # Find suitable range for tail fitting, and perform tail fit
                        peak_position = np.argmax(tcspc_y)
                        fit_start = np.uint64(peak_position + np.ceil(2E-9 / fixer.micro_time_resolution))
                        flat_background, _ = fixer.get_background_tail_fit(channels_spec_2, 
                                                                             peak_position, 
                                                                             fit_start,
                                                                             use_drift_correction = use_bleaching_correction,
                                                                             use_burst_removal = use_burst_removal,
                                                                             use_mse_filter = use_mse_filter,
                                                                             calling_function = script_name)
                        
                        # Get FLCS weights
                        _ = fixer.get_flcs_background_filter(tcspc_x, 
                                                               tcspc_y, 
                                                               flat_background, 
                                                               channels_spec_2,
                                                               calling_function = script_name)

                    # Correlate with filters up to this point applied
                    _ = fixer.get_correlation_uncertainty(channels_spec_1,
                                                            channels_spec_2, 
                                                            use_drift_correction = use_bleaching_correction,
                                                            use_flcs_bg_corr = use_flcs_bg_subtraction,
                                                            use_burst_removal = use_burst_removal,
                                                            use_mse_filter = use_mse_filter,
                                                            calling_function = script_name)
                    
                    # END of background correction block, and of entire processing of this channel combination
                    
            except:
                # If this channel combination failed, write that to log, and continue with next
                fixer.write_to_logfile(log_header = 'Error: Logging traceback.',
                                         log_message = traceback.format_exc(),
                                         calling_function = script_name)
                                         
                # pass # just skip if this channel combination failed
    except:
        # File failed. In this case, print error message to console, but go to next file
        traceback.print_exc()
        failed_path.extend([in_path])

print('Job done.')