#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:19:58 2023

@author: krohn
"""

# For data processing
import FCS_cleaner
import numpy as np
import tttrlib 

# For IO
import os

# Other
import matplotlib.pyplot as plt # Plotting
import warnings # To suppress pointless pyplot warnings
import datetime # For creating time stamps

test_dir = '../test_data/'
# test_file = 'Measurement1_T0s_1.ptu'
# test_file = 'Calibration1_AF488_A655_1.ptu'
test_file = 'Sample1_Mins_2uMD_2uME_spot_surface_1.ptu'
# test_file = 'Sample1_Mins_2uMD_2uME_spot_solution_1.ptu'
# test_file = '04 FCS lipid bilayer ATTO655.ptu'

in_path = os.path.join(test_dir, test_file)
out_path = os.path.join(test_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M") +'_'+ os.path.splitext(test_file)[0])

photon_data = tttrlib.TTTR(in_path,'PTU')

# channels_spec_1 = 0
channels_spec_1 = ((1,), ((0.5,), (1,)))
# channels_spec_2 = 1
channels_spec_2 = ((2,), ((0.5,), (1,)))

script_name = 'User/main'

#%% Import data
Cleaner = FCS_cleaner.FCS_cleaner(photon_data = photon_data, 
                                    out_path = out_path,
                                    tau_min = 1E-6,
                                    tau_max = 1.0,
                                    sampling = 8,
                                    cross_corr_symm = True,
                                    correlation_method = 'default',
                                    subtract_afterpulsing = False,
                                    afterpulsing_params_path = 'ap_params_D044.csv', # Default is dummy, must be specified if subtract_afterpulsing == True,
                                    weights_ext = None,
                                    write_results = True,
                                    include_header = True,
                                    write_log = True)
Cleaner.update_params()



channel_1, micro_time_gates_1 = Cleaner.check_channels_spec(channels_spec_1)
channel_2, micro_time_gates_2 = Cleaner.check_channels_spec(channels_spec_2)


# #%% Basic correlation operation simple_correlation()
# photon_inds1 = photon_data.get_selection_by_channel([0])
# photon_inds2 = photon_data.get_selection_by_channel([1])
# macrotimes1 = photon_data.macro_times[photon_inds1]
# macrotimes2 = photon_data.macro_times[photon_inds2]
# lags1, cc1 = Cleaner.simple_correlation(np.uint64(macrotimes1),
#                                         np.uint64(macrotimes2))

# # Plot
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.semilogx(lags1*1E-9, cc1-1, 'dg')
# ax.set_title('simple_correlation() output')
# with warnings.catch_warnings():
#     # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
#     warnings.simplefilter('ignore')
#     fig.show()


# #%% Correlation operation with basic settings but using the more flexible function correlation_apply_filters()
# lags, cc, acr1, acr2 = Cleaner.correlation_apply_filters(channels_spec_1, 
#                                                         channels_spec_2,
#                                                         ext_indices = np.array([]),
#                                                         tau_min = None, 
#                                                         tau_max = None,
#                                                         use_ext_weights = False, 
#                                                         use_detrend_weights = False, 
#                                                         remove_bursts = False,
#                                                         remove_anomalous_segments = False,
#                                                         suppress_logging = False,
#                                                         calling_function = script_name)

# # Plot
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.semilogx(lags*1E-9, cc, 'dg')
# ax.set_title('correlation_apply_filters() output')
# with warnings.catch_warnings():
#     # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
#     warnings.simplefilter('ignore')
#     fig.show()

# #%% get_segment_ccs() for multiple CFs from a single measurement
# lag_times, segment_ccs, usable_segments, start_stop = Cleaner.get_segment_ccs(channels_spec_1,
#                                                                             channels_spec_2,
#                                                                             minimum_window_length = Cleaner.acquisition_time / 10.,
#                                                                             tau_min = None,
#                                                                             tau_max = None,
#                                                                             use_ext_weights = False,
#                                                                             use_detrend_weights = False,
#                                                                             remove_bursts = False,
#                                                                             remove_anomalous_segments = False,
#                                                                             suppress_logging = False,
#                                                                             calling_function = script_name
#                                                                             )

# # PLot
# fig, ax = plt.subplots(nrows=1, ncols=1)
# [ax.semilogx(lags*1E-9, segment_ccs[:,i_segment], 'g') for i_segment in usable_segments]
# ax.set_title('Multile CFs from get_segment_ccs()')
# with warnings.catch_warnings():
#     # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
#     warnings.simplefilter('ignore')
#     fig.show()
    
# #%% get_Wohland_SD for uncertainty
# sd_cc_W = Cleaner.get_Wohland_SD(channels_spec_1,
#                                 channels_spec_2,
#                                 minimum_window_length = [],
#                                 tau_max = None,
#                                 tau_min = None,
#                                 use_ext_weights = False, 
#                                 use_detrend_weights = False, 
#                                 remove_bursts = False,
#                                 remove_anomalous_segments = False,
#                                 suppress_logging = False,
#                                 calling_function = script_name)

# # PLot
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.semilogx(lags*1E-9, cc, 'dg')
# ax.semilogx(lags*1E-9, cc+sd_cc_W, '-m')
# ax.semilogx(lags*1E-9, cc-sd_cc_W, '-m')
# ax.set_title('CF with get_Wohland_SD() uncertainty')
# with warnings.catch_warnings():
#     # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
#     warnings.simplefilter('ignore')
#     fig.show()

# #%% get_bootstrap_SD() for uncertainty
# sd_cc_bs = Cleaner.get_bootstrap_SD(channels_spec_1,
#                                     channels_spec_2, 
#                                     n_bootstrap_reps = 10,
#                                     tau_min = None,
#                                     tau_max = None,
#                                     use_ext_weights = False, 
#                                     use_detrend_weights = False,
#                                     remove_bursts = False,
#                                     remove_anomalous_segments = False,
#                                     suppress_logging = False,
#                                     calling_function = script_name)

# # Plot
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.semilogx(lags*1E-9, cc, 'dg')
# ax.semilogx(lags*1E-9, cc+sd_cc_bs, '-m')
# ax.semilogx(lags*1E-9, cc-sd_cc_bs, '-m')
# ax.set_title('CF with get_bootstrap_SD() uncertainty')
# with warnings.catch_warnings():
#     # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
#     warnings.simplefilter('ignore')
#     fig.show()
    
    
    
    
#%% get_correlation_uncertainty(): Shortcut for correlation + uncertainty + autosaving

lags, cc, sd_cc, acr1, acr2 = Cleaner.get_correlation_uncertainty(channels_spec_1,
                                                                    channels_spec_2, 
                                                                    default_uncertainty_method = 'Wohland',
                                                                    minimum_window_length = [],
                                                                    n_bootstrap_reps = 10,
                                                                    tau_min = None,
                                                                    tau_max = None,
                                                                    use_ext_weights = False, 
                                                                    use_detrend_weights = False,
                                                                    remove_bursts = False,
                                                                    remove_anomalous_segments = False,
                                                                    suppress_logging = False,
                                                                    calling_function = script_name)

# fcs_fit = FCS_cleaner.G_diff_3dim_1comp(tau = lags*1E-9, 
#                                         G = cc, 
#                                         sigma_G = sd_cc, 
#                                         count_rate = 1.0, 
#                                         BG = 0., 
#                                         PSF_radius = 0.2, 
#                                         PSF_aspect_ratio = 5., 
#                                         initial_params = {'N':1., 'tau diffusion':1E-4, 'offset':0.})
fcs_fit = FCS_cleaner.G_diff_3dim_2comp(tau = lags*1E-9, 
                                        G = cc, 
                                        sigma_G = sd_cc, 
                                        count_rate = 1.0, 
                                        BG = 0., 
                                        PSF_radius = 0.2, 
                                        PSF_aspect_ratio = 5., 
                                        initial_params = {'N':1., 
                                                          'tau diffusion 1':1E-4, 
                                                          'tau diffusion 2':1E-2, 
                                                          'f1': 0.5,
                                                          'offset':0.})
fit_results = fcs_fit.run_fit()

# PLot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogx(lags*1E-9, cc, 'dg')
ax.semilogx(lags*1E-9, cc+sd_cc, '-m')
ax.semilogx(lags*1E-9, cc-sd_cc, '-m')
ax.semilogx(lags*1E-9, fit_results['G_prediction'], 'k')

ax.set_title('CF with uncertainty from single shortcut call')
with warnings.catch_warnings():
    # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
    warnings.simplefilter('ignore')
    fig.show()



#%% Time trace generation and automated choice of bin width

time_trace_sampling = Cleaner.get_trace_time_scale(channels_spec_1,
                                                    min_avg_counts = 10.0,
                                                    min_bin_width = 1E-4,
                                                    use_tau_diff = True,
                                                    ext_indices = np.array([]),
                                                    use_ext_weights = False,
                                                    use_detrend_weights = False,
                                                    remove_bursts = False,
                                                    remove_anomalous_segments = False,
                                                    suppress_logging = False,
                                                    calling_function = script_name)

time_trace_counts_1, time_trace_t_1 = Cleaner.get_time_trace(channels_spec_1,
                                                                time_trace_sampling,
                                                                ext_indices = np.array([]),
                                                                use_ext_weights = False,
                                                                use_detrend_weights = False,
                                                                remove_bursts = False,
                                                                remove_anomalous_segments = False,
                                                                suppress_logging = False,
                                                                calling_function = script_name)

if channels_spec_1 != channels_spec_2:
    time_trace_counts_2, time_trace_t_2 = Cleaner.get_time_trace(channels_spec_2,
                                                                time_trace_sampling,
                                                                ext_indices = np.array([]),
                                                                use_ext_weights = False,
                                                                use_detrend_weights = False,
                                                                remove_bursts = False,
                                                                remove_anomalous_segments = False,
                                                                suppress_logging = False,
                                                                calling_function = script_name)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(time_trace_t_1, time_trace_counts_1, 'g', label = 'first channel')

if channels_spec_1 != channels_spec_2:
    ax.plot(time_trace_t_2, time_trace_counts_2, 'm', label = 'second channel')
    ax.legend()
    
ax.set_title('get_time_trace() output')

with warnings.catch_warnings():
    # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
    warnings.simplefilter('ignore')
    fig.show()

#%% Burst detection 1 - Two-step procedure with threshold tuning followed by threshold application

# auto_threshold_1 = Cleaner.get_auto_threshold(time_trace_counts_1,
#                                                 threshold_alpha = 0.02,
#                                                 suppress_logging = False,
#                                                 calling_function = script_name)

# binarized_trace_1 = Cleaner.threshold_trace(time_trace_counts_1, 
#                                             threshold_counts = auto_threshold_1,
#                                             suppress_logging = False,
#                                             calling_function = script_name)

# if channels_spec_1 != channels_spec_2:

#     auto_threshold_2 = Cleaner.get_auto_threshold(time_trace_counts_2,
#                                                     threshold_alpha = 0.02,
#                                                     suppress_logging = False,
#                                                     calling_function = script_name)
    
#     binarized_trace_2 = Cleaner.threshold_trace(time_trace_counts_2, 
#                                                   threshold_counts = auto_threshold_2,
#                                                   suppress_logging = False,
#                                                   calling_function = script_name)

# # Plot
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.plot(time_trace_t_1, time_trace_counts_1, 'g', label = 'first channel')
# ax.plot([time_trace_t_1[0], time_trace_t_1[-1]], [auto_threshold_1, auto_threshold_1], '-y')
# ax.plot(time_trace_t_1[binarized_trace_1], time_trace_counts_1[binarized_trace_1], 'oy')

# if channels_spec_1 != channels_spec_2:
#     ax.plot(time_trace_t_2, time_trace_counts_2, 'm', label = 'second channel')
#     ax.plot([time_trace_t_2[0], time_trace_t_2[-1]], [auto_threshold_2, auto_threshold_2], '-r')
#     ax.plot(time_trace_t_2[binarized_trace_2], time_trace_counts_2[binarized_trace_2], 'or')
#     ax.legend()
    
# ax.set_title('Time traces with threshold (get_auto_threshold() and threshold_trace())')

# with warnings.catch_warnings():
#     # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
#     warnings.simplefilter('ignore')
#     fig.show()
    
    


#%% Burst detection 2 - One-step procedure handling automated threshold implicitly

# # Channel 1
# binarized_trace_1 = Cleaner.threshold_trace(time_trace_counts_1, 
#                                             threshold_alpha = 0.02,
#                                             threshold_counts = None,
#                                             suppress_logging = False,
#                                             calling_function = script_name)

# if channels_spec_1 != channels_spec_2:

#     # Channel 2
#     binarized_trace_2 = Cleaner.threshold_trace(time_trace_counts_2, 
#                                                   threshold_alpha = 0.02,
#                                                   threshold_counts = None,
#                                                   suppress_logging = False,
#                                                   calling_function = script_name)

# # Plot
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.plot(time_trace_t_1, time_trace_counts_1, 'g', label = 'first channel')
# ax.plot([time_trace_t_1[0], time_trace_t_1[-1]], [auto_threshold_1, auto_threshold_1], '-y')
# ax.plot(time_trace_t_1[binarized_trace_1], time_trace_counts_1[binarized_trace_1], 'oy')

# if channels_spec_1 != channels_spec_2:
#     ax.plot(time_trace_t_2, time_trace_counts_2, 'm', label = 'second channel')
#     ax.plot([time_trace_t_2[0], time_trace_t_2[-1]], [auto_threshold_2, auto_threshold_2], '-r')
#     ax.plot(time_trace_t_2[binarized_trace_2], time_trace_counts_2[binarized_trace_2], 'or')
#     ax.legend()
    
# ax.set_title('Time traces with threshold (one-step threshold_trace())')

# with warnings.catch_warnings():
#     # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
#     warnings.simplefilter('ignore')
#     fig.show()

#%% Burst removal for burst-cleaned correlation

# # We just performed burst annotation in two channels independently. 
# # Which one to use? Here are some examples of what one can do:

# # # Remove time points labelled as "burst" IN CHANNEL 1, ignoring channel 2
# # burst_bins = binarized_trace_1 

# # # Remove time points labelled as "burst" IN CHANNEL 2, ignoring channel 1
# # burst_bins = binarized_trace_2

# # Remove all time points labelled as "burst" IN AT LEAST ONE OF THE TWO channels
# burst_bins = np.logical_or(binarized_trace_1, binarized_trace_2) 

# # # Remove only time points labelled as "burst" IN BOTH CHANNELS
# # burst_bins = np.logical_and(binarized_trace_1, binarized_trace_2) 

# # It is also possible to go back perform burst detection using the SUM of two or more channels:
# # burst_bins = Cleaner.threshold_trace(time_trace_counts_1 + time_trace_counts_2, 
# #                                     threshold_alpha = 0.02,
# #                                     threshold_counts = None,
# #                                     suppress_logging = False)    
    
# # You can even use channels for burst detection that are not even included in 
# # the correlation function calculation, or create a binarized traces from some 
# # totally different custom calculation.

# # Remove bursts from both channels
# photon_is_burst = Cleaner.update_photons_from_bursts(burst_bins,
#                                                      time_trace_sampling,
#                                                      update_weights = True,
#                                                      update_macro_times = True,
#                                                      suppress_logging = False,
#                                                      calling_function = script_name)

#%% remove_bursts: Shortcut call for get_auto_threshold + threshold_trace + logical combination of traces from multiple channels + update_photons_from_bursts

# Multi-channels data is concatenated along axis 1
time_traces = np.concatenate((time_trace_counts_1.reshape((time_trace_counts_1.shape[0], 1)), time_trace_counts_2.reshape((time_trace_counts_2.shape[0], 1))), axis = 1) if channels_spec_1 != channels_spec_2 else time_trace_counts_1.reshape((time_trace_counts_1.shape[0], 1))

burst_bins, photon_is_burst = Cleaner.run_burst_removal(time_traces,
                                                        time_trace_sampling,
                                                        multi_channel_handling = 'OR',
                                                        threshold_alpha = 0.02,
                                                        threshold_counts = None,
                                                        update_weights = True,
                                                        update_macro_times = True,
                                                        suppress_logging = False,
                                                        calling_function = script_name)



#%% Look at what burst removal did to the correlation function

# Burst removal applied
lags_br, cc_br, sd_cc_br, acr1_br, acr2_br = Cleaner.get_correlation_uncertainty(channels_spec_1,
                                                                                channels_spec_2, 
                                                                                default_uncertainty_method = 'Wohland',
                                                                                minimum_window_length = [],
                                                                                n_bootstrap_reps = 10,
                                                                                tau_min = None,
                                                                                tau_max = None,
                                                                                use_ext_weights = False, 
                                                                                use_detrend_weights = False,
                                                                                remove_bursts = True,
                                                                                remove_anomalous_segments = False,
                                                                                suppress_logging = False,
                                                                                calling_function = script_name)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogx(lags*1E-9, cc, 'dg', label = 'Raw')
ax.semilogx(lags_br*1E-9, cc_br, '.m', label = 'Burst removal')
ax.set_title('CFs with and without burst removal')
ax.legend()
with warnings.catch_warnings():
    # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
    warnings.simplefilter('ignore')
    fig.show()
    
    
    

#%% Re-plot time traces with burst removal
time_trace_counts_br_1, time_trace_t_1 = Cleaner.get_time_trace(channels_spec_1,
                                                                time_trace_sampling,
                                                                ext_indices = np.array([]),
                                                                use_ext_weights = False,
                                                                use_detrend_weights = False,
                                                                remove_bursts = True,
                                                                remove_anomalous_segments = False,
                                                                suppress_logging = False,
                                                                calling_function = script_name)

if channels_spec_1 != channels_spec_2:
    time_trace_counts_br_2, time_trace_t_2 = Cleaner.get_time_trace(channels_spec_2,
                                                                    time_trace_sampling,
                                                                    ext_indices = np.array([]),
                                                                    use_ext_weights = False,
                                                                    use_detrend_weights = False,
                                                                    remove_bursts = True,
                                                                    remove_anomalous_segments = False,
                                                                    suppress_logging = False,
                                                                    calling_function = script_name)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(time_trace_t_1, time_trace_counts_br_1, 'g', label = 'first channel')

if channels_spec_1 != channels_spec_2:
    ax.plot(time_trace_t_2, time_trace_counts_br_2, 'm', label = 'second channel')
    ax.legend()

ax.set_title('Time traces with burst removal')
with warnings.catch_warnings():
    # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
    warnings.simplefilter('ignore')
    fig.show()


#%% Bleaching/drift correction



# Fit and perform correction

# Channel 1
poly_params_1 , time_trace_poly_1, time_trace_detrend_1 = Cleaner.polynomial_detrending_rss(time_trace_counts_br_1, 
                                                                                            time_trace_t_1, 
                                                                                            channels_spec_1,
                                                                                            detrend_order = None,
                                                                                            max_detrend_order = 10,
                                                                                            update_detrend_weights = True,
                                                                                            ext_indices = np.array([]),
                                                                                            use_ext_weights = False,
                                                                                            remove_bursts = True,
                                                                                            remove_anomalous_segments=False,
                                                                                            suppress_logging = False)

if channels_spec_1 != channels_spec_2:

    # Channel 2
    poly_params_2 , time_trace_poly_2, time_trace_detrend_2 = Cleaner.polynomial_detrending_rss(time_trace_counts_br_2, 
                                                                                                time_trace_t_2, 
                                                                                                channels_spec_2,
                                                                                                detrend_order = None,
                                                                                                max_detrend_order = 10,
                                                                                                update_detrend_weights = True,
                                                                                                ext_indices = np.array([]),
                                                                                                use_ext_weights = False,
                                                                                                remove_bursts = True,
                                                                                                remove_anomalous_segments=False,
                                                                                                suppress_logging = False)

    # Plot time traces
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(time_trace_t_1*1E-9, time_trace_counts_br_1, 'g', label = 'Burst removal')
    ax[0].plot(time_trace_t_1*1E-9, time_trace_detrend_1, 'm', label = 'Burst removal and detrend')
    ax[0].plot(time_trace_t_1*1E-9, time_trace_poly_1, 'y', label = 'Polynomial fit of order '+ str(len(poly_params_1)-1))
    ax[1].plot(time_trace_t_2*1E-9, time_trace_counts_br_2, 'g', label = 'Burst removal')
    ax[1].plot(time_trace_t_2*1E-9, time_trace_detrend_2, 'm', label = 'Burst removal and detrend')
    ax[1].plot(time_trace_t_2*1E-9, time_trace_poly_2, 'y', label = 'Polynomial fit of order '+ str(len(poly_params_2)-1))
    ax[1].legend()
    ax[0].set_title('Time traces with and without detrending')
    
else:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(time_trace_t_1*1E-9, time_trace_counts_br_1, 'g', label = 'Burst removal')
    ax.plot(time_trace_t_1*1E-9, time_trace_detrend_1, 'm', label = 'Burst removal and detrend')
    ax.plot(time_trace_t_1*1E-9, time_trace_poly_1, 'y', label = 'Polynomial fit of order '+ str(len(poly_params_1)-1))
    ax.legend()
    ax.set_title('Time traces with and without detrending')

    
with warnings.catch_warnings():
    # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
    warnings.simplefilter('ignore')
    fig.show()
    
    

#%% Look at what detrending did to the correlation function

# Burst removal and detrend applied
lags_br_dt, cc_br_dt, sd_cc_br_dt, acr1_br_dt, acr2_br_dt = Cleaner.get_correlation_uncertainty(channels_spec_1,
                                                                                channels_spec_2, 
                                                                                default_uncertainty_method = 'Wohland',
                                                                                minimum_window_length = [],
                                                                                n_bootstrap_reps = 10,
                                                                                tau_min = None,
                                                                                tau_max = None,
                                                                                use_ext_weights = False, 
                                                                                use_detrend_weights = True,
                                                                                remove_bursts = True,
                                                                                remove_anomalous_segments = False,
                                                                                suppress_logging = False,
                                                                                calling_function = script_name)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogx(lags_br*1E-9, cc_br, 'dg', label = 'Burst removal')
ax.semilogx(lags_br_dt*1E-9, cc_br_dt, '.m', label = 'Burst removal and detrend')
ax.set_title('CFs with and without bleaching/drift correction')
ax.legend()
with warnings.catch_warnings():
    # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
    warnings.simplefilter('ignore')
    fig.show()
    
    
    
#%% Remove measurement segments with anomalous correlation function

# While this filter can obviously be applied in any way you like, its intended 
# use this module is mostly to correct remaining artifacts that burst removal 
# and detrending were unable to correct. Hence here we use it on the already
# burst- and drift/bleaching-corrected data.

# This is a one-step function call, no need to call two or three functions in sequence
mse_matrix, good_segments = Cleaner.discard_anomalous_segments(channels_spec_1,
                                                                channels_spec_2,
                                                                minimum_window_length = [], # Automatic, will try to split into 10 segments
                                                                anomaly_threshold = 2.5,
                                                                ignore_amplitude_fluctuations = True,
                                                                update_macro_times = True,
                                                                update_weights = True,
                                                                tau_max = None,
                                                                tau_min = 1E5, # 100 us to cut off excessive noise
                                                                use_ext_weights = False, 
                                                                use_detrend_weights = True, 
                                                                remove_bursts = True,
                                                                suppress_logging = False)

# Look at what that did to the correlation function

# Burst removal, detrend, and anomalous segment removal applied
lags_br_dt_ar, cc_br_dt_ar, sd_cc_br_dt_ar, acr1_br_dt_ar, acr2_br_dt_ar = Cleaner.get_correlation_uncertainty(channels_spec_1,
                                                                                                                channels_spec_2, 
                                                                                                                default_uncertainty_method = 'Wohland',
                                                                                                                minimum_window_length = [],
                                                                                                                n_bootstrap_reps = 10,
                                                                                                                tau_min = None,
                                                                                                                tau_max = None,
                                                                                                                use_ext_weights = False, 
                                                                                                                use_detrend_weights = True,
                                                                                                                remove_bursts = True,
                                                                                                                remove_anomalous_segments = True,
                                                                                                                suppress_logging = False,
                                                                                                                calling_function = script_name)
# Plot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogx(lags_br_dt*1E-9, cc_br_dt, 'dg', label = 'Burst removal and detrend')
ax.semilogx(lags_br_dt_ar*1E-9, cc_br_dt_ar, '.m', label = 'Burst rem., detrend, and anom. CF removal')
ax.set_title('CFs with and without removal of anomalous segments')
ax.legend()
with warnings.catch_warnings():
    # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
    warnings.simplefilter('ignore')
    fig.show()
    
    
    
#%% Fit IRF to get some info on how to fit the background
irf_file = 'Calibration7_KI_fluor_A655_1.ptu'
irf_path = os.path.join(test_dir, irf_file)

irf_TTTR = tttrlib.TTTR(irf_path,'PTU')
irf_peak_center_1, irf_peak_fwhm_1, irf_fit_1 = Cleaner.find_IRF_position(channels_spec_1,
                                                                      irf_TTTR,
                                                                      suppress_logging = False,
                                                                      calling_function = script_name)




fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(irf_fit_1.x, irf_fit_1.y, 'g', label='Data 1st channel')
ax.semilogy(irf_fit_1.x, irf_fit_1.prediction, 'y', alpha = 0.7, label='Fit 1st channel')

if channels_spec_1 != channels_spec_2:

    irf_peak_center_2, irf_peak_fwhm_2, irf_fit_2 = Cleaner.find_IRF_position(channels_spec_2,
                                                                          irf_TTTR,
                                                                          suppress_logging = False,
                                                                          calling_function = script_name)

    ax.semilogy(irf_fit_2.x, irf_fit_2.y, 'm', label='Data 2nd channel')
    ax.semilogy(irf_fit_2.x, irf_fit_2.prediction, 'r', alpha = 0.7, label='Fit 2nd channel')
    ax.legend()

fig.supxlabel('TCSPC bin')
fig.supylabel('Counts')
ax.set_title('IRF with Gaussian approximation')

with warnings.catch_warnings():
    # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
    warnings.simplefilter('ignore')
    fig.show()
    
    
#%% Fit background level
flat_background_1, tail_fit_1 = Cleaner.get_background_tail_fit(channels_spec_1,
                                                                irf_peak_center = irf_peak_center_1,
                                                                fit_start = np.floor(irf_peak_center_1 + 3. * irf_peak_fwhm_1).astype(np.uint64),
                                                                ext_indices = np.array([]),
                                                                use_ext_weights = False,
                                                                use_detrend_weights = True,
                                                                remove_bursts = True,
                                                                remove_anomalous_segments = True,
                                                                suppress_logging = False,
                                                                calling_function = script_name)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(tail_fit_1.x, tail_fit_1.y, 'g', label='Data 1st channel')
ax.semilogy(tail_fit_1.x, tail_fit_1.prediction, 'y', alpha = 0.7, label='Fit 1st channel')
ax.semilogy([tail_fit_1.x[0], tail_fit_1.x[-1]], [flat_background_1, flat_background_1], ':y', label='Background 1st channel')

if channels_spec_1 != channels_spec_2:

    flat_background_2, tail_fit_2 = Cleaner.get_background_tail_fit(channels_spec_2,
                                                                    irf_peak_center = irf_peak_center_2,
                                                                    fit_start = np.floor(irf_peak_center_2 + 3. * irf_peak_fwhm_2).astype(np.uint64),
                                                                    ext_indices = np.array([]),
                                                                    use_ext_weights = False,
                                                                    use_detrend_weights = True,
                                                                    remove_bursts = True,
                                                                    remove_anomalous_segments = True,
                                                                    suppress_logging = False,
                                                                    calling_function = script_name)

    ax.semilogy(tail_fit_2.x, tail_fit_2.y, 'm', label='Data 2nd channel')
    ax.semilogy(tail_fit_2.x, tail_fit_2.prediction, 'r', alpha = 0.7, label='Fit 2nd channel')
    ax.semilogy([tail_fit_2.x[0], tail_fit_2.x[-1]], [flat_background_2, flat_background_2], ':r', label='Background 2nd channel')

    ax.legend()

fig.supxlabel('TCSPC bin')
fig.supylabel('Counts')
ax.set_title('Decay tail fit')

with warnings.catch_warnings():
    # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
    warnings.simplefilter('ignore')
    fig.show()
    
    
    
    
#%% FLCS via FCS_cleaner methods


tcspc_x_1, tcspc_y_1 = Cleaner.get_tcspc_histogram(channels_spec_1,
                                                    micro_times = [],
                                                    ext_indices = np.array([]),
                                                    use_ext_weights = False,
                                                    use_detrend_weights = True,
                                                    remove_bursts = True,
                                                    remove_anomalous_segments = True,
                                                    suppress_logging = False,
                                                    calling_function = script_name)

patterns_norm_full_1, flcs_weights_full_1 = Cleaner.get_flcs_background_filter(tcspc_x_1,
                                                                               tcspc_y_1,
                                                                               flat_background = flat_background_1,
                                                                               channels_spec = channels_spec_1,
                                                                               handle_outside = 'zero',
                                                                               update_weights = True,
                                                                               ext_indices = np.array([]),
                                                                               suppress_logging = False,
                                                                               calling_function = script_name)

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].semilogy(np.arange(patterns_norm_full_1.shape[0]), patterns_norm_full_1[:,0], 'g', label='Data')
ax[0].semilogy(np.arange(patterns_norm_full_1.shape[0]), patterns_norm_full_1[:,1], 'y', label='Background')

ax[1].plot(np.arange(patterns_norm_full_1.shape[0]), flcs_weights_full_1[:,0], 'g', label = 'Signal')
ax[1].plot(np.arange(patterns_norm_full_1.shape[0]), flcs_weights_full_1[:,1], 'y', label = 'Background')

if channels_spec_1 != channels_spec_2:
    
    tcspc_x_2, tcspc_y_2 = Cleaner.get_tcspc_histogram(channels_spec_2,
                                                        micro_times = [],
                                                        ext_indices = np.array([]),
                                                        use_ext_weights = False,
                                                        use_detrend_weights = True,
                                                        remove_bursts = True,
                                                        remove_anomalous_segments = True,
                                                        suppress_logging = False,
                                                        calling_function = script_name)
    
    patterns_norm_full_2, flcs_weights_full_2 = Cleaner.get_flcs_background_filter(tcspc_x_2,
                                                                                   tcspc_y_2,
                                                                                   flat_background = flat_background_2,
                                                                                   channels_spec = channels_spec_2,
                                                                                   handle_outside = 'zero',
                                                                                   update_weights = True,
                                                                                   ext_indices = np.array([]),
                                                                                   suppress_logging = False,
                                                                                   calling_function = script_name)
    
    ax[0].semilogy(np.arange(patterns_norm_full_2.shape[0]), patterns_norm_full_2[:,0], 'm', alpha = 0.7, label='Data')
    ax[0].semilogy(np.arange(patterns_norm_full_2.shape[0]), patterns_norm_full_2[:,1], 'r', alpha = 0.7, label='Background')

    ax[1].plot(np.arange(patterns_norm_full_2.shape[0]), flcs_weights_full_2[:,0], 'm', alpha = 0.7, label = 'Signal')
    ax[1].plot(np.arange(patterns_norm_full_2.shape[0]), flcs_weights_full_2[:,1], 'r', alpha = 0.7, label = 'Background')

ax[0].legend()
ax[1].legend()


# Burst removal, detrend, and anomalous segment removal applied
lags_br_dt_ar_bg, cc_br_dt_ar_bg, sd_cc_br_dt_ar_bg, acr1_br_dt_ar_bg, acr2_br_dt_ar_bg= Cleaner.get_correlation_uncertainty(channels_spec_1,
                                                                                                                channels_spec_2, 
                                                                                                                default_uncertainty_method = 'Wohland',
                                                                                                                minimum_window_length = [],
                                                                                                                n_bootstrap_reps = 10,
                                                                                                                tau_min = None,
                                                                                                                tau_max = None,
                                                                                                                use_ext_weights = False, 
                                                                                                                use_detrend_weights = True,
                                                                                                                use_flcs_bg_corr = True,
                                                                                                                remove_bursts = True,
                                                                                                                remove_anomalous_segments = True,
                                                                                                                suppress_logging = False,
                                                                                                                calling_function = script_name)
# Plot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogx(lags_br_dt_ar*1E-9, cc_br_dt_ar, '.g', label = 'Burst rem., detrend, and anom. CF removal')
ax.semilogx(lags_br_dt_ar_bg*1E-9, cc_br_dt_ar_bg, '.m', label = 'Burst rem., detrend, anom. CF rem., bg corr.')
ax.set_title('CFs with and without background correction')
ax.legend()
with warnings.catch_warnings():
    # Suppress Spyder throwing a ton of warnings about matplotlib although everything works fine actually.
    warnings.simplefilter('ignore')
    fig.show()

