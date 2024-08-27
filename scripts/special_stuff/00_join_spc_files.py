# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:31:05 2024

@author: Krohn
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:19:58 2023

@author: krohn
"""


'''
This script was used for some data from Don Lamb's lab. Here, a series of 
short acquisitions was performed, but I want to have a single long one for 
analysis. So this script concatenates them for processing in FCS_Fixer.

Note that this script as things stand uses methods of the tttrlib.TTTR class that
are not available in tttrlib versions that work with the rest of FCS_Fixer 
(FCS_Fixer uses an older version!). That's the whole reason why this is a 
distinct script rather than part of FCS_Fixer. 

Unfortunately the changes I would have to make to FCS_Fixer to adapt 
it to newer tttrlib versions are somewhat deep-sitting, 
so I am not too enthusiatic about making them as long as I can find workarounds.

'''


# For I/O
import os
import glob


# Processing
import numpy as np
import tttrlib

#%% Input data
# You can extend dir_names with as many source directories as you want
dir_names=[]
dir_names.extend([r'some_base_dir/NP_in_buffer_time_0'])


#%% Go through diretories and find all .spc files, creating effectively groupings of directory, file, time stamp, channel index for each of them


_file_names = []
_dir_names = []
time_stamps = []
channel_indices = []
for dir_name in dir_names:

    for file_name in glob.glob(os.path.join(dir_name, '*.spc')):
        _file_names.extend([os.path.split(file_name)[1]])
        _dir_names.extend([os.path.split(file_name)[0]])
        time_stamps.append(int(os.path.splitext(file_name)[0][-6:-2]))
        channel_indices.append(int(os.path.splitext(file_name)[0][-1]))
time_stamps = np.array(time_stamps)
channel_indices = np.array(channel_indices)




#%% Associate files

unique_dirs = []
for directory in _dir_names:
    if not (directory in unique_dirs):
        unique_dirs.extend([directory])

for unique_dir in unique_dirs:
    
    for channel in np.unique(channel_indices):
        
        # Some initialization
        init_done = False
        macro_time_offset = 0
        
        for time_stamp in np.unique(time_stamps):
            
            # Iterate over all time stamps
            for i_file, file_name in enumerate(_file_names):
                # Look for each file if it belongs to this file group and time stamp
                if time_stamps[i_file] == time_stamp and _dir_names[i_file] == unique_dir and channel_indices[i_file] == channel:
                    
                    # Correct time stamp: Load and add to data
                    if not init_done:
                        # First file - create TTTR object
                        photon_data = tttrlib.TTTR(os.path.join(unique_dir, 
                                                                file_name),
                                                   'SPC-130')
                        out_file_name = os.path.splitext(file_name)[0][:-6]
                        init_done = True
        
                    else:
                        # We have TTTR object already - append
                        new_photons = tttrlib.TTTR(os.path.join(unique_dir, 
                                                                file_name),
                                                   'SPC-130')
                        photon_data.append_events(macro_times = new_photons.macro_times,
                                                  micro_times = new_photons.micro_times,
                                                  routing_channels = np.ones_like(new_photons.macro_times, dtype = np.int8) * channel_indices[i_file],
                                                  event_types = np.ones_like(new_photons.macro_times, dtype = np.int8),
                                                  shift_macro_time = False,
                                                  macro_time_offset = macro_time_offset)
                    print(f'Channel: {channel}, Time stamp: {time_stamp} \n Number of photons: {photon_data.macro_times.shape[0]} \n last photon macro time: {photon_data.macro_times.max()} \n channels used: {photon_data.get_used_routing_channels()}' )
                    
            macro_time_offset = np.max(photon_data.macro_times)
        
        # All files that we could get for this group combined - write file (SPC-130 format)
        photon_data_sort = photon_data[np.argsort(photon_data.macro_times)]
        _ = photon_data_sort.write(os.path.join(unique_dir,
                                                out_file_name + f'_ch{channel}.spc'))

