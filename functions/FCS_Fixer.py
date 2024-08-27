# Data processing
import tttrlib # TTTR data-specific procedures
import numpy as np # General data manipulation

# Curve fitting
from lmfit import minimize, Parameters 
from scipy.optimize import minimize as sp_minimize 

# Specific distributions we need
from scipy.special import erfcinv 
from scipy.stats import f as f_dist

# For I/O and log writing
import os # Mostly file naming
import datetime # Time stamps
import matplotlib.pyplot as plt # Plotting
import pandas as pd # exporting tables as .csv
from itertools import cycle # used only in plotting
import glob

# misc
import warnings # For suppressing expectable but pointless warnings
import traceback # For logging some specific exceptions caught exceptions
import multiprocessing # For running simple processing pipelines in a parallel fashion


#%% Small helper structure

def isint(object_to_check):
    '''
    Just a shorthand to check if the object in question is one out of many 
    int types, which we use in other functions.

    Parameters
    ----------
    object_to_check : 
        Some object whose type we want to check.

    Returns
    -------
    Bool
        True if the object is an int type, else False.

    '''
    return type(object_to_check) in [int, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]


def isiterable(object_to_check):
    '''
    Just a shorthand to check if the object in question is one out of a few 
    iterable types, which we use in other functions.

    Parameters
    ----------
    object_to_check : 
        Some object whose type we want to check.

    Returns
    -------
    Bool
        True if the object is an allowed iterable type, else False.

    '''
    return type(object_to_check) in [list, tuple, np.ndarray]


def isfloat(object_to_check):
    '''
    Just a shorthand to check if the object in question is one out of a few 
    float types, which we use in other functions.

    Parameters
    ----------
    object_to_check : 
        Some object whose type we want to check.

    Returns
    -------
    Bool
        True if the object is an allowed float type, else False.

    '''
    return type(object_to_check) in [float, np.float16, np.float32, np.float64]

def lin_scaling_of_data(data_raw,
                        data_ref):
    '''
    Linearly scale a dataset (data_raw) to be of comparable 
    amplitude to another (data_ref). Used as helper function to 
    self.cc_mse_filter_get_mse().

    Parameters
    ----------
    data_raw : np.ndarray
        Dataset to be scaled. Must be of same shape as data1
    data_ref : np.ndarray
        Dataset that serves as reference for scaling.
        
    Returns
    -------
    data_scaled : np.ndarray
        Scaled version of data_raw.

    '''
    fit_res = sp_minimize(lambda a: np.sum((data_ref - a * data_raw)**2), 
                           1, 
                           options = {'maxiter':10}).x[0]
    data_scaled = data_raw * fit_res
        
    return data_scaled
    
    
class Parallel_scheduler():
    # Relatively simple class wrapping FCS_Fixer to run the 
    # "standard pipeline" using parallel processing
    
    def __init__(self,
                 in_paths,
                 tau_min = 1E-6,
                 tau_max = 1.,  
                 sampling = 6,
                 correlation_method = 'default',
                 cross_corr_symm = False,
                 use_calibrated_AP_subtraction = False,
                 afterpulsing_params_path = '',
                 list_of_channel_pairs = [],
                 use_burst_removal = False,
                 use_drift_correction = False,
                 use_mse_filter = False,
                 use_flcs_bg_corr = False,
                 default_uncertainty_method = 'Wohland',
                 write_intermediate_ccs = False,
                 write_pcmh = True,
                 out_dir = ''
                 ):
        '''
        
        Set up the global settings for parallel processing

        Parameters
        ----------
        in_paths : 
            List of paths to raw data.
        tau_min, tau_max : TYPE, optional
            OPTIONAL Floats. Minimum and maximum lag time of correlation in seconds, 
            respectively, with defaults 1E-6 and 1.
        sampling : 
            OPTIONAL Int with default 6. Density of sampling of correlation function.
        correlation_method : 
            OPTIONAL string with defaut 'default', alternative 'lamb'. Denotes 
            which correlation function calculation algorithm to use (passed into
            tttrlib.Correlator() class).
        cross_corr_symm : 
            OPTIONAL bool with default False. In case of cross-correlation function
            calculation, should we assume time symmetry? Doing so increases 
            signal-to-noise ratio by averaging forward and backward cross
            correlation. Auto-correlations are not affected.
        use_calibrated_AP_subtraction : 
            OPTIONAL bool with default False. Whether to use calibrated
            afterpulsing subtraction (unused for cross-correlation and for
            auto-correlations if FLCS background subtraction is used.)
        afterpulsing_params_path :
            OPTIONAL string/path with default '' (empty). Path to afterpulsing calibration
            file. Necessary if use_calibrated_AP_subtraction == True, otherwise 
            ignored.
        list_of_channel_pairs :
            OPTIONAL iterable of 2-element iterables of channels_spec tuples, with syntax 
            as delivered by FCS_Fixer.get_channel_combinations(). Specifies which 
            correlation operations to perform. If left empty, the software will perform all
            possible auto- and cross-correlations between channels that have a somewhat 
            reasonable number of photons in the raw data.
        use_burst_removal : 
            OPTIONAL bool with default False. Whether to use burst removal filter.
        use_drift_correction : 
            OPTIONAL bool with default False. Whether to use bleaching/drift correction
        use_mse_filter : 
            OPTIONAL bool with default False. Whether to use MSE-based removal
            of measurement time segments with anomalous correlation function.
        use_flcs_bg_corr : 
            OPTIONAL bool with default False. Whether to use FLCS to remove 
            laser-independent background.
        default_uncertainty_method : 
            OPTIONAL string with default 'Wohland'. Alternative is 'Bootstrap'.
            Choice of uncertainty calculation method to be used by FCS_Fixer.get_correlation_uncertainty()
            If 'Wohland' is chosen, FCS_Fixer.get_Wohland_SD() is used as the default
            method of standard deviation calculation, and FCS_Fixer.get_bootstrap_SD()
            as the backup method. If 'Bootstrap' is chosen, the software will 
            directly go to FCS_Fixer.get_bootstrap_SD().
        write_intermediate_ccs :
            OPTIONAL bool with default False. Whether or not to write intermediate
            FCS output at every filtering step.
        write_pcmh :
            OPTIONAL bool with default True. Whether to add PC(M)H export to pipeline.
        out_dir :
            OPTIONAL string with empty str as default. If empty, a subfolder 
            will be created next to the input file. If a dir is given, the 
            software will instead create a directory in out_dir that mirrors 
            the last up to 3 layers of directory hierarchy in in_path and place 
            the output there - convenient for collecting output from multiple 
            input experiments.

        '''
        
        self.in_paths = in_paths
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.sampling = sampling
        self.correlation_method = correlation_method
        self.cross_corr_symm = cross_corr_symm
        self.use_calibrated_AP_subtraction = use_calibrated_AP_subtraction
        self.afterpulsing_params_path = afterpulsing_params_path
        self.list_of_channel_pairs = list_of_channel_pairs
        self.use_burst_removal = use_burst_removal
        self.use_drift_correction = use_drift_correction
        self.use_mse_filter = use_mse_filter
        self.use_flcs_bg_corr = use_flcs_bg_corr
        self.default_uncertainty_method = default_uncertainty_method
        self.write_intermediate_ccs = write_intermediate_ccs
        self.write_pcmh = write_pcmh
        self.out_dir = out_dir
        
        
        
    @staticmethod
    def run_standard_pipeline_all_channels(in_path,
                                           tau_min,
                                           tau_max, 
                                           sampling,
                                           use_calibrated_AP_subtraction,
                                           use_burst_removal,
                                           use_drift_correction,
                                           use_mse_filter,
                                           use_flcs_bg_corr,
                                           default_uncertainty_method = 'Wohland',
                                           write_intermediate_ccs = False,
                                           write_pcmh = True,
                                           afterpulsing_params_path = '',
                                           list_of_channel_pairs = [],
                                           cross_corr_symm = False,
                                           correlation_method = 'default',
                                           out_dir = '',
                                           job_name = ''):
        '''
        Load a file into FCS_Fixer, and run the standard pipeline in all
        channel combinations available.

        Parameters
        ----------
        in_path : 
            Path to raw data.
        tau_min, tau_max : 
            Floats. Minimum and maximum lag time of correlation in seconds, 
            respectively.
        sampling : 
            Int. Density of sampling of correlation function.
        use_calibrated_AP_subtraction : 
            Bool. Whether to use calibrated afterpulsing subtraction (unused
            for cross-correlation and for auto-correlations if FLCS background 
            subtraction is used.)
        use_burst_removal : 
            Bool. Whether to use burst removal filter.
        use_drift_correction : 
            Bool. Whether to use bleaching/drift correction.
        use_mse_filter : 
            Bool. Whether to use MSE-based removal of measurement time segments 
            with anomalous correlation function.
        use_flcs_bg_corr : 
            Bool. Whether to use FLCS to remove laser-independent background.
        default_uncertainty_method : 
            OPTIONAL string with default 'Wohland'. Alternative is 'Bootstrap'.
            Choice of uncertainty calculation method to be used by FCS_Fixer.get_correlation_uncertainty()
            If 'Wohland' is chosen, FCS_Fixer.get_Wohland_SD() is used as the default
            method of standard deviation calculation, and FCS_Fixer.get_bootstrap_SD()
            as the backup method. If 'Bootstrap' is chosen, the software will 
            directly go to FCS_Fixer.get_bootstrap_SD().
        afterpulsing_params_path :
            OPTIONAL string/path with default '' (empty). Path to afterpulsing calibration
            file. Necessary if use_calibrated_AP_subtraction == True, otherwise 
            ignored.
        write_intermediate_ccs :
            OPTIONAL bool with default False. Whether or not to write intermediate
            FCS output at every filtering step.
        write_pcmh :
            OPTIONAL bool with default True. Whether to add PC(M)H export to pipeline.
        list_of_channel_pairs :
            OPTIONAL iterable of 2-element iterables of channels_spec tuples, with syntax 
            as delivered by FCS_Fixer.get_channel_combinations(). Specifies which 
            correlation operations to perform. If left empty, the software will perform all
            possible auto- and cross-correlations between channels that have a somewhat 
            reasonable number of photons in the raw data.
        cross_corr_symm : 
            OPTIONAL bool with default False. In case of cross-correlation function
            calculation, should we assume time symmetry? Doing so increases 
            signal-to-noise ratio by averaging forward and backward cross
            correlation. Auto-correlations are not affected.
        correlation_method : 
            OPTIONAL string with defaut 'default', alternative 'lamb'. Denotes 
            which correlation function calculation algorithm to use (passed into
            tttrlib.Correlator() class).
        out_dir :
            OPTIONAL string with empty str as default. If empty, a subfolder 
            will be created next to the input file. If a dir is given, the 
            software will instead create a directory in out_dir that mirrors 
            the last up to 3 layers of directory hierarchy in in_path and place 
            the output there - convenient for collecting output from multiple 
            input experiments.
        job_name : 
            OPTIONAL string with default '' (empty). Passed into 
            FCS_Fixer.write_to_log() as the "calling_function" name from 
            which high-level methods are being called (for logging).


        '''
        if os.path.splitext(in_path)[1] == '.ptu':
            photon_data = tttrlib.TTTR(in_path,'PTU')
            
            
        elif os.path.splitext(in_path)[1] == '.spc':
            photon_data = tttrlib.TTTR(in_path,'SPC-130')

            
        in_dir, in_file = os.path.split(in_path)
        out_name_common = os.path.splitext(in_file)[0]
        
        if out_dir == '':
            out_path = os.path.join(in_dir, out_name_common + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M"))
            
        elif type(out_dir) == str:
            # We have an excplicit out_dir to use
            
            # First sequentially split the in_dir to mirror up to 3 hierarchy levels
            dir_levels = 0
            try:
                tmpdir, out_dir1 = os.path.split(in_dir)
                dir_levels += 1
            except:
                pass
            
            try:
                tmpdir, out_dir2 = os.path.split(tmpdir)
                dir_levels += 1
            except:
                pass
            
            try:
                _, out_dir3 = os.path.split(tmpdir)
                dir_levels += 1
            except:
                pass
            
            if dir_levels == 3:
                out_dir = os.path.join(out_dir, out_dir3)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                    
            if dir_levels >= 2:
                out_dir = os.path.join(out_dir, out_dir2)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

            if dir_levels >= 1:
                out_dir = os.path.join(out_dir, out_dir1)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                
            # Complete out_path
            out_path = os.path.join(out_dir, out_name_common + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M"))

        fixer = FCS_Fixer(photon_data = photon_data, 
                           out_path = out_path,
                           tau_min = tau_min,
                           tau_max = tau_max,
                           sampling = sampling,
                           cross_corr_symm = cross_corr_symm,
                           correlation_method = correlation_method,
                           subtract_afterpulsing = use_calibrated_AP_subtraction,
                           afterpulsing_params_path = afterpulsing_params_path,
                           write_results = True,
                           include_header = False,
                           write_log = True)
        fixer.update_params()
        
        
        # Auto-detect all channels in the file and enumerate all combinations to correlate, if not specified
        if list_of_channel_pairs == []:
            list_of_channel_pairs = fixer.get_channel_combinations(min_photons = 1000)

        # Perform all correlations
        for channels_spec_1, channels_spec_2 in list_of_channel_pairs:
            
            try: 
                fixer.run_standard_pipeline(channels_spec_1,
                                            channels_spec_2,
                                            use_burst_removal,
                                            use_drift_correction,
                                            use_mse_filter,
                                            use_flcs_bg_corr,
                                            default_uncertainty_method = default_uncertainty_method,
                                            write_intermediate_ccs = write_intermediate_ccs,
                                            write_pcmh = write_pcmh,
                                            calling_function = job_name)

            except:
                # If this channel combination failed, write that to log, and continue with next
                fixer.write_to_logfile(log_header = 'Error: Logging traceback.',
                                         log_message = traceback.format_exc(),
                                         calling_function = job_name)
                                         
                # pass # just skip if this channel combination failed


    def par_process_wrapper(self,
                            process_id):
        '''
        Uses the process_id to identify a file and start its processing

        Parameters
        ----------
        process_id : 
            Int. Index of file to process within self.in_paths list, which also 
            serves as process ID.

        '''

        print(f'[{process_id}] Processing ' + self.in_paths[process_id] + '...')

        self.run_standard_pipeline_all_channels(in_path = self.in_paths[process_id],
                                                tau_min = self.tau_min,
                                                tau_max = self.tau_max, 
                                                sampling = self.sampling,
                                                use_calibrated_AP_subtraction = self.use_calibrated_AP_subtraction,
                                                use_burst_removal = self.use_burst_removal,
                                                use_drift_correction = self.use_drift_correction,
                                                use_mse_filter = self.use_mse_filter,
                                                use_flcs_bg_corr = self.use_flcs_bg_corr,
                                                default_uncertainty_method = self.default_uncertainty_method,
                                                write_intermediate_ccs = self.write_intermediate_ccs,
                                                write_pcmh = self.write_pcmh,
                                                afterpulsing_params_path = self.afterpulsing_params_path,
                                                list_of_channel_pairs = self.list_of_channel_pairs,
                                                cross_corr_symm = self.cross_corr_symm,
                                                correlation_method = self.correlation_method,
                                                out_dir = self.out_dir,
                                                job_name = f'process_{process_id}')

        
    def run_parallel_processing(self,
                                process_count):
        '''
        Execute assign and parallel jobs.

        Parameters
        ----------
        process_count : 
            Int. Number of parallel processing cores to use.


        '''
        # Create pool
        pool = multiprocessing.Pool(processes = process_count)
        
        try:
            # Parallel processing wrapper
            pool.map(self.par_process_wrapper, np.arange(len(self.in_paths)))
        
        except:
            # Something went wrong - whatever, not too much we can do
            traceback.print_exc()

        finally:
            # In any case, close parpool at the end
            pool.close()


    
class G_diff_3dim_1comp():
    # Object-based fitting of single-component 3D diffusion autocorrelation FCS model.
    
    def __init__(self, 
                tau, 
                G, 
                sigma_G = 1.0, 
                count_rate = 1.0, 
                BG = 0., 
                PSF_radius = 0.2, 
                PSF_aspect_ratio = 5., 
                initial_params = {'N':1., 'tau diffusion':1E-4, 'offset':0.}):
        '''
        For fitting a single component 3D diffusion model to a correlation function

        Parameters
        ----------
        tau : np.array
            Lag times for fit (independent variable), typically assumed to be in s.
        G : np.array
            Correlation function to fit (dependent variable).
        sigma_G : np.array, optional
            Uncertainty of correlation function, must be of same length as 
            tau and G. The default is 1.0, which translates to no weighting
        count_rate : scalar, optional
            Count rate of fluorescence signal (background-free). Only needed if 
            correction of correlation amplitude for background is desired, 
            otherwise can be ignored. The default is 1.0 (dummy value).
        BG : scalar, optional
            Count rate of background contribution. Only needed if 
            correction of correlation amplitude for background is desired, 
            otherwise can be ignored. The default is 0.0 (dummy value).
        PSF_radius : Float, optional
            1/e^2 Width of the detection volume element in the xy plane, typically in 
            micrometers. The default is 0.2.
        PSF_aspect_ratio : Float, optional
            Aspect ratio of the detection volume element, i.e., ratio of 1/e^2
            width in z over PSF_radius. The default is 5.
        initial_params : dict of floats, optional
            Dict that defines the initial parameters of the two fitted 
            variables N and tau_diffusion. The default is 
            {'N':1., 'tau diffusion':1E-4, 'offset':0.}. You need not specify 
            all key-values pairs.


        '''
        
        # Correlation function
        self.tau = tau
        self.G = G
        
        if np.all(sigma_G == 1.0):
            self.sigma_G = np.ones_like(G)# Dummy
        else:
            self.sigma_G = sigma_G # Use actual input
        
        # Calibration/metadata, although these can often be left at dummy values
        self.count_rate = count_rate
        self.BG = BG
        self.PSF_aspect_ratio = PSF_aspect_ratio
        self.PSF_radius = PSF_radius

        # Set up parameters in optimization problem
        self.fit_params = Parameters()
        
        self.fit_params.add('N', 
                            value = initial_params['N'] if 'N' in initial_params.keys() else 1., 
                            min = 0., 
                            vary = True)
            
        self.fit_params.add('tau_D', 
                            value = initial_params['tau diffusion'] if 'tau diffusion' in initial_params.keys() else 1E-4, 
                            min = np.min(self.tau), 
                            vary = True)
            
        self.fit_params.add('offset', 
                            value = initial_params['offset'] if 'offset' in initial_params.keys() else 0., 
                            vary = True)
            


    def g3ddiff1comp_expression(self, g0, tau_D, aspect_ratio, offset):
        '''
        Actual expression to fit/predict.

        Parameters
        ----------
        g0 : Float
            Correlation amplitude.
        tau_D : Float
            Diffusion time.
        aspect_ratio : Float
            Aspect ratio of the detection volume element, i.e., ratio of 1/e^2
            width in z over PSF_radius. 
        offset : Float
            Correlation function offset at long times.
        
        Returns
        -------
        np.array
            Predicted (model) correlation function.

        '''
        # Actual expression to fit
        g_diff = (1+self.tau/tau_D)**(-1) / (np.sqrt(1 + aspect_ratio**(-2) * self.tau/tau_D))
        
        return g0 * g_diff + offset


    def g3ddiff1comp_fun(self, params):
        '''
        Brings together the model parameters to be fitted (N and tau_diffusion)
        with fixed parameters (stored as attributes) to calculate the full 
        correlation fucntion model using child function self.g3ddiff1comp_expression()

        Parameters
        ----------
        params : dict: {'N':Float, 'tau_D': Float, 'offset': Float} 
            Model parameters.

        Returns
        -------
        np.array
            Predicted (model) correlation function including amplitude corrections.

        '''
        
        # Unpack fit params
        N = params['N'].value
        tau_D = params['tau_D'].value
        offset = params['offset'].value
        
        # Calculate amplitude based on fit parameter N and a few fixed parameters
        BG_correction = ((self.count_rate / (self.count_rate + self.BG)) ** 2)
        g0 = BG_correction/(2 * np.sqrt(2) * N)
        
        return self.g3ddiff1comp_expression(g0, tau_D, self.PSF_aspect_ratio, offset)


    def g3ddiff1comp_residual(self, params):
        '''
        Wraps self.g3ddiff1comp_fun() to compare the model with current 
        parameters to data stored in attributes and return a weighted residuals 

        Parameters
        ----------
        params : dict: {'N':Float, 'tau_D': Float, 'offset', Float} 
            Model parameters.

        Returns
        -------
        np.array
            Weighted resiudals (not squared, not sum!).

        '''
        # Wrapper for g3ddiff1comp_fun() that is the target for minimization
        residual = self.G - self.g3ddiff1comp_fun(params)
        
        return residual / self.sigma_G


    def run_fit(self):
        '''
        Wraps everything above to simply run the fit and return a dict with all 
        relevant results.

        Returns
        -------
        return_dict : Dict containing floats and np.arrays
            Fit results including model parameters, simple derived parameters,
            the best-fit correlation function model, and fit statistics.

        '''
        # Actual fitting
        result = minimize(self.g3ddiff1comp_residual, 
                            self.fit_params, 
                            method='nelder')
        
        fitted_params = result.params

        # Residuals and goodness-of-fit        
        G_prediction = self.g3ddiff1comp_fun(fitted_params)
        r = self.G - G_prediction
        weighted_r = r/self.sigma_G
        
        chi_squared = np.sum((weighted_r)**2)
        n_lags = len(self.tau)
        
        red_chi_squared = chi_squared/(n_lags-result.nvarys)
        BIC = n_lags*np.log(red_chi_squared/n_lags) + result.nvarys*np.log(n_lags)

        # Recalculate fit parameters
        N_fitted = fitted_params['N'].value
        tau_D_fitted = fitted_params['tau_D'].value
        offset_fitted = fitted_params['offset'].value
        D_fitted = (self.PSF_radius ** 2) / (4 * tau_D_fitted)
        
        dtau_D_fitted =fitted_params['tau_D'].stderr
        try: 
            dD_fitted = D_fitted * dtau_D_fitted/tau_D_fitted 
        except:
            dD_fitted = np.nan
            
        CPP_peak = self.count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))

        # Return
        return_dict = {'PSF radius': self.PSF_radius, 
                        'PSF aspect ratio': self.PSF_aspect_ratio, 
                        'N': N_fitted,
                        'D': D_fitted, 
                        'dD': dD_fitted, 
                        'Tau diffusion': tau_D_fitted, 
                        'offset': offset_fitted,
                        'CPP average': CPP_avg, 
                        'CPP peak': CPP_peak, 
                        'Chi squared': red_chi_squared, 
                        'r': r,
                        'weighted_r':weighted_r,
                        'G_prediction': G_prediction, 
                        'Count Rate': self.count_rate, 
                        'BIC': BIC}

        return return_dict


class G_diff_3dim_2comp():
    
    def __init__(self, 
                tau, 
                G, 
                sigma_G = 1.0, 
                count_rate = 1.0, 
                BG = 0., 
                PSF_radius = 0.2, 
                PSF_aspect_ratio = 5., 
                initial_params = {'N':1., 
                                  'tau diffusion 1':1E-4, 
                                  'tau diffusion 2':1E-2, 
                                  'f1': 0.5,
                                  'offset':0.}):
        '''
        For fitting a 2-component 3D diffusion model to a correlation function

        Parameters
        ----------
        tau : np.array
            Lag times for fit (independent variable), typically assumed to be in s.
        G : np.array
            Correlation function to fit (dependent variable).
        sigma_G : np.array, optional
            Uncertainty of correlation function, must be of same length as 
            tau and G. The default is 1.0, which translates to no weighting
        count_rate : scalar, optional
            Count rate of fluorescence signal (background-free). Only needed if 
            correction of correlation amplitude for background is desired, 
            otherwise can be ignored. The default is 1.0 (dummy value).
        BG : scalar, optional
            Count rate of background contribution. Only needed if 
            correction of correlation amplitude for background is desired, 
            otherwise can be ignored. The default is 0.0 (dummy value).
        PSF_radius : Float, optional
            1/e^2 Width of the detection volume element in the xy plane, typically in 
            micrometers. The default is 0.2.
        PSF_aspect_ratio : Float, optional
            Aspect ratio of the detection volume element, i.e., ratio of 1/e^2
            width in z over PSF_radius. The default is 5.
        initial_params : dict of floats, optional
            Dict that defines the initial parameters of the 4 fitted 
            variables (total N, fast fraction, and fast and slow diffusion times). 
            The default is {'N':1., 'tau diffusion 1':1E-4, 'tau diffusion 2':1E-2, 'f1': 0.5, 'offset':0.}.
            You need not specify all key-values pairs.
            
        Returns
        -------
        None.
        '''
        
        # Correlation function
        self.tau = tau
        self.G = G
        
        if np.all(sigma_G == 1.0):
            self.sigma_G = np.ones_like(G)# Dummy
        else:
            self.sigma_G = sigma_G # Use actual input
        
        # Calibration/metadata, although these can often be left at dummy values
        self.count_rate = count_rate
        self.BG = BG
        self.PSF_aspect_ratio = PSF_aspect_ratio
        self.PSF_radius = PSF_radius

        # Set up parameters in optimization problem
        self.fit_params = Parameters()
        
        self.fit_params.add('N', 
                            value = initial_params['N'] if 'N' in initial_params.keys() else 1., 
                            min = 0., 
                            vary = True)
        
        self.fit_params.add('tau_D1', 
                            value = initial_params['tau diffusion 1'] if 'tau diffusion 1' in initial_params.keys() else 1E-4, 
                            min = np.min(self.tau), 
                            vary = True)

        self.fit_params.add('rho_D', 
                            value = (initial_params['tau diffusion 2'] if 'tau diffusion 2' in initial_params.keys() else 1E-2) / self.fit_params['tau_D1'].value , 
                            min = 1.,
                            vary = True)  # rho_D = tau_D2/tau_D1
            
        self.fit_params.add('tau_D2', 
                            expr ='tau_D1 * rho_D', 
                            vary = False)  # with rho_D > 1, ensures tau_D2 > tau_D1

        self.fit_params.add('f1', 
                            value = initial_params['f1'] if 'f1' in initial_params.keys() else 0.5, 
                            min = 0.,
                            max = 1.,
                            vary = True)
            
        self.fit_params.add('f2', 
                            expr ='1-f1', 
                            vary = False)

        self.fit_params.add('offset', 
                            value = initial_params['offset'] if 'offset' in initial_params.keys() else 0., 
                            vary = True)
            

    def g3ddiff2comp_expression(self, g0, f1, tau_D1, f2, tau_D2, aspect_ratio, offset):
        '''
        Actual expression to fit/predict.
        
        Parameters
        ----------
        g0 : Float
            Correlation amplitude.
        f1 : Float.
            Fast fraction
        tau_D1 : Float
            Fast diffusion time.
        f2 : Float.
            Slow fraction.
        tau_D2 : Float
            Slow diffusion time.
        aspect_ratio : Float
            Aspect ratio of the detection volume element, i.e., ratio of 1/e^2
            width in z over PSF_radius. 
        offset : Float
            Correlation function offset at long lag times.
            
        Returns
        -------
        np.array
            Predicted (model) correlation function.
        '''
        
        G_Diff1 = f1  * (1+self.tau/tau_D1)**(-1) / (np.sqrt(1 + aspect_ratio**(-2) * self.tau/tau_D1))
        G_Diff2 = f2  * (1 + self.tau / tau_D2) ** (-1) / (np.sqrt(1 + aspect_ratio ** (-2) * self.tau / tau_D2))
        
        return g0 * (G_Diff1 + G_Diff2) + offset


    def g3ddiff2comp_fun(self, params):
        '''
        Wrapper to g3ddiff2comp_expression() that basically handles the 
        distinction between fit parameters and fixed parameters. 
        
        Parameters
        ----------
        params : dict: {'N':Float, 'tau_D1':Float, 'tau_D2':Float, 'f1': Float, 'f2': Float, 'offset': Float} 
            Model parameters.

        Returns
        -------
        np.array
            Predicted (model) correlation function including amplitude corrections.

        '''
        
        # Unpack fit params
        N = params['N'].value
        tau_D1 = params['tau_D1'].value
        tau_D2 = params['tau_D2'].value
        f1 = params['f1'].value
        f2 = params['f2'].value
        offset = params['offset'].value
        
        # Calculate amplitude based on fit parameter N and a few fixed parameters
        BG_correction = ((self.count_rate / (self.count_rate + self.BG)) ** 2)
        g0 = BG_correction/(2 * np.sqrt(2) * N)
        
        return self.g3ddiff2comp_expression(g0, f1, tau_D1, f2, tau_D2, self.PSF_aspect_ratio, offset)


    def g3ddiff2comp_residual(self, params):
        '''
        Wraps self.g3ddiff2comp_fun() to compare the model with current 
        parameters to data stored in attributes and return a weighted residuals 

        Parameters
        ----------
        params : dict: {'N':Float, 'tau_D1':Float, 'tau_D2':Float, 'f1': Float, 'f2': Float, 'offset': Float} 
            Model parameters.

        Returns
        -------
        np.array
            Weighted resiudals (not squared, not sum!).

        '''
        
        # Wrapper for g3ddiff2comp_fun() that is the target for minimization
        residual = self.G - self.g3ddiff2comp_fun(params)
        
        return residual / self.sigma_G


    def run_fit(self):
        '''
        Wraps the above methods to simply run the fit and return a dict with all 
        relevant results.

        Returns
        -------
        return_dict : Dict containing floats and np.arrays
            Fit results including model parameters, simple derived parameters,
            the best-fit correlation function model, and fit statistics.
        '''
        
        # Actual fitting
        result = minimize(self.g3ddiff2comp_residual, 
                            self.fit_params, 
                            method='nelder')
        
        fitted_params = result.params

        # Residuals and goodness-of-fit        
        G_prediction = self.g3ddiff2comp_fun(fitted_params)
        r = self.G - G_prediction
        weighted_r = r/self.sigma_G
        
        chi_squared = np.sum((weighted_r)**2)
        n_lags = len(self.tau)
        
        red_chi_squared = chi_squared/(n_lags-result.nvarys)
        BIC = n_lags*np.log(red_chi_squared/n_lags) + result.nvarys*np.log(n_lags)

        # Recalculate fit parameters
        N_fitted = fitted_params['N'].value
        offset_fitted = fitted_params['offset'].value

        tau_D1_fitted = fitted_params['tau_D1'].value
        D1_fitted = (self.PSF_radius ** 2) / (4 * tau_D1_fitted)
        f1_fitted = fitted_params['f1'].value
        
        tau_D2_fitted = fitted_params['tau_D2'].value
        D2_fitted = (self.PSF_radius ** 2) / (4 * tau_D2_fitted)
        f2_fitted = fitted_params['f2'].value

        dtau_D1_fitted =fitted_params['tau_D1'].stderr
        try: 
            dD1_fitted = D1_fitted * dtau_D1_fitted/tau_D1_fitted
        except:
            dD1_fitted = np.nan
            
        dtau_D2_fitted =fitted_params['tau_D2'].stderr
        try: 
            dD2_fitted = D2_fitted * dtau_D2_fitted/tau_D2_fitted
        except:
            dD2_fitted = np.nan
            
        CPP_peak = self.count_rate / N_fitted
        CPP_avg = CPP_peak / (2 * np.sqrt(2))

        # Return
        return_dict = {'PSF radius': self.PSF_radius, 
                        'PSF aspect ratio': self.PSF_aspect_ratio, 
                        'N': N_fitted,
                        'D1': D1_fitted, 
                        'dD1': dD1_fitted, 
                        'Tau diffusion 1': tau_D1_fitted, 
                        'f1': f1_fitted,
                        'D2': D2_fitted, 
                        'dD2': dD2_fitted, 
                        'Tau diffusion 2': tau_D2_fitted, 
                        'f2': f2_fitted,
                        'offset': offset_fitted,
                        'CPP average': CPP_avg, 
                        'CPP peak': CPP_peak, 
                        'Chi squared': red_chi_squared, 
                        'r': r,
                        'weighted_r':weighted_r,
                        'G_prediction': G_prediction, 
                        'Count Rate': self.count_rate, 
                        'BIC': BIC}

        return return_dict
            
    
class Polynomial_fit():

    def __init__(self, 
                time, 
                counts, 
                poly_order):
        '''
        In essence mimicks np.polynomial.polynomial.polyfit(), but with the 
        one single important difference that the zero order term is constrained 
        to be non-negative.

        Parameters
        ----------
        time : 
            np.array. Independent variable of fit.
        counts : 
            np.array. Dependent variable of fit; also used to create weighting 
            function based on poisson statistics.
        poly_order : 
            Degree of the polynomial to fit.


        '''
        # Data to fit
        self.time = time
        self.counts = counts
        
        # Polynomial order
        self.poly_order = poly_order
        
        # Raising the time tags to the power of the polynomial orders is a 
        # repetitive calculation that's always the same, so we do that just once in the beginning
        self.time_power = self.time.reshape((self.time.shape[0], 1)) ** np.arange(poly_order+1).reshape((1, poly_order+1))
        
        sigma_counts = np.sqrt(counts)
        sigma_counts[sigma_counts == 0] = np.max(sigma_counts) * 1E3
        self.sigma_counts = sigma_counts
        
        # Set up parameters in optimization problem
        self.fit_params = Parameters()
        
        # Zero order term is > 0, in fact, we say it must be >= the smallest nonzero value in counts
        self.fit_params.add('c0', 
                            value = self.counts.mean(), 
                            min = np.min(self.counts[self.counts > 0]), 
                            vary = True)
        
        if self.poly_order > 0:
            for order in range(1, self.poly_order+1):
                # Higher order terms are unconstrained
                self.fit_params.add(f'c{order}', 
                                    value = 0., 
                                    vary = True)


    def polynomial_expression(self, coefficient_array):
        '''
        Actual expression to fit.

        Parameters
        ----------
        coefficient_array : 
            np.array of coefficients of polynomial.

        Returns
        -------
        prediction : 
            np.array with predicted polynomial model.

        '''
        prediction = (self.time_power * coefficient_array).sum(axis = 1)
                
        return prediction


    def polynomial_fun(self, coefficients):
        '''
        Wrapper translating between the formatting of coefficients that lmfit wants,
        and the formatting that self.polynomial_expression() wants.

        Parameters
        ----------
        coefficients : 
            fit parameter dict from lmfit.

        Returns
        -------
        prediction : 
            np.array with predicted polynomial model.

        '''
        coefficient_array = np.zeros((1, self.poly_order+1,), dtype = np.float64)
        coefficient_array[0, 0] = coefficients['c0']
        
        if self.poly_order > 0:
            for order in range(1, self.poly_order+1):
                coefficient_array[0, order] = coefficients[f'c{order}']
                
        return self.polynomial_expression(coefficient_array)
        

    def polynomial_residual(self, coefficients):
        '''
        Wrapper for self.polynomial_fun() that is the target for minimization

        Parameters
        ----------
        coefficients : 
            fit parameter dict from lmfit.

        Returns
        -------
        weighted_residuals
            np.array with residuals of current prediction weighted by uncertainties.

        '''
        # Wrapper for g3ddiff1comp_fun() that is the target for minimization
        residual = self.counts - self.polynomial_expression(coefficients)
        
        return residual / self.sigma_counts


    def run_fit(self):
        '''
        Wraps everything above to simply run the fit and return a dict with all 
        relevant results.

        Returns
        -------
        return_dict : Dict containing floats and np.arrays
            Fit results including model parameters, simple derived parameters,
            the best-fit correlation function model, and fit statistics.

        '''
        # Actual fitting
        result = minimize(self.polynomial_residual, 
                            self.fit_params, 
                            method='nelder')
        
        fitted_params = result.params

        # Residuals and goodness-of-fit        
        prediction = self.polynomial_expression(fitted_params)
        chi_squared = np.sum(((self.counts - prediction) /self.sigma_counts)**2)
        
        n_data_points = self.counts.shape[0]
        red_chi_squared = chi_squared/(n_data_points-result.nvarys)
        
        # Return
        fit_params = np.zeros((self.poly_order+1,), dtype = np.float64)
        fit_params[0] = fitted_params['c0'].value
        
        if self.poly_order > 0:
            for order in range(1, self.poly_order+1):
                fit_params[order] = fitted_params[f'c{order}'].value
                        
        return fit_params, red_chi_squared
    
    
class TCSPC_quick_fit():
    
    def __init__(self,
                 x_data, 
                 y_data, 
                 model,
                 initial_params = {'x_0': 0.,
                                   'y_0': 0.,
                                   'amp': 1000.,
                                   'gauss_fwhm': 1.,
                                   'exp_tau': 1.}):
        '''
        For simple fits to TCSPC data, bunding two distinct models (IRF 
        localization and exponential tail fitting) into a single class

        Parameters
        ----------
        x_data : np.array
            TCSPC bin time labels (independent variable).
        y_data : np.array
            TCSPC photon counts (dependent variable in fit, also used to calculate
            weights under Poisson count assumption).
        model : string: 'gauss' or 'exponential'
            Which of the two built-in models to use. Use 'gauss' e.g. for estimating 
            position and width of the IRF, and 'exponential' for a simple 
            monoexponential tail fit.
        initial_params : dict, optional
            Initial parameters of fit. The default is {'x_0': 0.,                                   
                                         'y_0': 0.,                                   
                                         'amp': 1000.,                                   
                                         'gauss_fwhm': 1.,                                   
                                         'exp_tau': 1.}.
            Note that initial_params['x_0'] is an initial parameter that is 
            optimized in fitting for the 'gauss' model, but used as a fixed 
            parameter that must be known a priori for the 'exponential' model.
            You need not specify all key-value pairs.

        Returns
        -------
        None.

        '''
        self.x = x_data
        self.y = y_data
        
        sigma = np.sqrt(y_data)
        sigma[sigma == 0] = 1E3 * np.max(sigma) # Replace zeros in inverse weighting function with large values (low weight)
        self.sigma = sigma
        
        # Set parameters dict for fitting
        self.fit_params = Parameters()
        
        self.fit_params.add('y_0', 
                            value = initial_params['y_0'] if 'y_0' in initial_params.keys() else 0., 
                            min = .0, 
                            vary = True)
        
        self.fit_params.add('amp', 
                            value = initial_params['amp'] if 'amp' in initial_params.keys() else 1000., 
                            min=0, 
                            vary=True)

        self.model = model
        if self.model == 'gauss':
            self.fit_params.add('x_0', 
                                value = initial_params['x_0'] if 'x_0' in initial_params.keys() else 0., 
                                min = 0., 
                                vary = True)
            self.fit_params.add('gauss_fwhm', 
                                value = initial_params['gauss_fwhm'] if 'gauss_fwhm' in initial_params.keys() else 1., 
                                min = 0., 
                                vary = True)
            
        elif self.model == 'exponential':
            self.fit_params.add('x_0', 
                                value = initial_params['x_0'] if 'x_0' in initial_params.keys() else 0., 
                                min = 0., 
                                vary = False) # Must be fixed here
            self.fit_params.add('exp_tau', 
                                value = initial_params['exp_tau'] if 'exp_tau' in initial_params.keys() else 1., 
                                min = 0., 
                                vary = True)
            
        else:
            raise ValueError('Invalid model. Use "gauss" or "exponential".')
        
        
    def expression_gauss(self, x_0, y_0, amp, gauss_fwhm):
        '''
        Actual expression to fit for the 'gauss' model

        Parameters
        ----------
        x_0 : Float
            Peak position of the Gaussian.
        y_0 : Float
            Offset in counts (i.e., laser-independent background).
        amp : Float
            Amplitude of the Gaussian.
        gauss_fwhm : Float
            Full width at half-maximum of the Gaussian.

        Returns
        -------
        np.array
            Predicted Gaussian model function.

        '''

        return  y_0 + amp/(gauss_fwhm*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(self.x-x_0)**2/gauss_fwhm**2)
    
    
    def residual_gauss(self, params):
        '''
        Wraps self.expression_gauss() to compare model with current parameters
        to data. Returns weighted residuals as minimization target.

        Parameters
        ----------
        params : dict: {'x_0': Float, 'y_0': Float, 'amp': Float, 'gauss_fwhm': Float}
            Dict of estimated model parameters. See self.expression_gauss().

        Returns
        -------
        np.array
            Weighted residuals given current parameters (NOT sum, NOT squared!!!).

        '''

        x_0 = params['x_0'].value
        y_0 = params['y_0'].value
        amp = params['amp'].value
        gauss_fwhm = params['gauss_fwhm'].value
        
        prediction = self.expression_gauss(x_0, y_0, amp, gauss_fwhm)
        
        return (self.y - prediction) / self.sigma


    def expression_exponential(self, x_0, y_0, amp, exp_tau):
        '''
        Actual expression to fit for the 'exponential' model

        Parameters
        ----------
        x_0 : Float
            Peak position of the TCSPC decay.
        y_0 : Float
            Offset in counts (i.e., laser-independent background).
        amp : Float
            Amplitude of the exponential decay, read out at t = x_0.
        exp_tau : Float
            1/e decay time of exponential decay.

        Returns
        -------
        np.array
            Predicted exponential model function.

        '''
        return  y_0 + amp * np.exp(-(self.x - x_0) / exp_tau)
    
    
    def residual_exponential(self, params):
        '''
        Wraps self.expression_exponential() to compare model with current parameters
        to data. Returns weighted residuals as minimization target.

        Parameters
        ----------
        params : dict: {'x_0': Float, 'y_0': Float, 'amp': Float, 'exp_tau': Float}
            Dict of estimated model parameters. See self.expression_exponential().

        Returns
        -------
        np.array
            Weighted residuals given current parameters (NOT sum, NOT squared!!!).

        '''

        # Wrapper for expression_exponential() that is the target for minimization
        x_0 = params['x_0'].value
        y_0 = params['y_0'].value
        amp = params['amp'].value
        exp_tau = params['exp_tau'].value
        
        prediction = self.expression_exponential(x_0, y_0, amp, exp_tau)
        
        return (self.y - prediction) / self.sigma


    def run_fit(self):
        '''
        Wraps the above methods to simply run the fit and return a dict with all 
        relevant results.

        Returns
        -------
        return_dict : Dict containing floats and np.arrays
            Fit results including model parameters and fit statistics.
        '''
        
        # Actual fitting
        if self.model == 'gauss':
            
            result = minimize(self.residual_gauss, 
                                self.fit_params, 
                                method='nelder')
            
            self.fitted_params = result.params
            
            self.prediction = self.expression_gauss(self.fitted_params['x_0'].value,
                                                     self.fitted_params['y_0'].value,
                                                     self.fitted_params['amp'].value,
                                                     self.fitted_params['gauss_fwhm'].value)
            
            red_chi_sq = np.sum(((self.y - self.prediction)/self.sigma)**2)/(len(self.x)-result.nvarys)
            
            # Return
            return_dict = {'x_0': self.fitted_params['x_0'].value,
                           'y_0': self.fitted_params['y_0'].value,
                           'amp': self.fitted_params['amp'].value,
                           'gauss_fwhm': self.fitted_params['gauss_fwhm'].value,
                           'red_chi_sq': red_chi_sq}
            
        elif self.model == 'exponential':
            
            with warnings.catch_warnings():
                # Suppress division-by-zero warnings that are frequently thrown at this point, 
                # although so far I had no instances where this really seems to 
                # have caused problems.
                warnings.simplefilter('ignore')
                result = minimize(self.residual_exponential, 
                                    self.fit_params, 
                                    method='nelder')
            
            self.fitted_params = result.params
            
            self.prediction = self.expression_exponential(self.fitted_params['x_0'].value,
                                                         self.fitted_params['y_0'].value,
                                                         self.fitted_params['amp'].value,
                                                         self.fitted_params['exp_tau'].value)
            
            red_chi_sq = np.sum(((self.y - self.prediction)/self.sigma)**2)/(len(self.x)-result.nvarys)
            
            # Return
            return_dict = {'x_0': self.fitted_params['x_0'].value,
                           'y_0': self.fitted_params['y_0'].value,
                           'amp': self.fitted_params['amp'].value,
                           'exp_tau': self.fitted_params['exp_tau'].value,
                           'red_chi_sq': red_chi_sq}
            
        else:
            raise ValueError('Invalid model. Use "gauss" or "exponential".')
            
        return return_dict






#%% Apply various corrections to tttr data and export correlation functions and time traces, with varyious degrees of automation possible


class FCS_Fixer():
    ### Class attributes
    __DATE = '2023-09-11'
    

    #%% Static methods
    # Basically, these are some methods that do not make much sense outside 
    # the contect of this class, but on the other hand have no use for any 
    # parameters accessed via self, so I might as well make them in principle 
    # accessible independent of class instance creation.
    
    @staticmethod
    def build_channels_spec(channels_indices,
                            micro_time_gates = None):
        '''
        
        Construct a channels_spec tuple from more intuitive input.

        Parameters
        ----------
        channels_indices:
            int or iterable of int. Routing channel (spectral/polarization/...) 
            indices to use.
        micro_time_gates:
            OPTIONAL interable of float. Specifies micro time gating to apply, 
            as positive gates of "use these photons and not the others".
            You can concatenate as many gates as you want. If left empty, no 
            gating is applied and all photons are used.
            Float must all be >= 0 and <= 1, specifying RELATIVE cutoffs along 
            the micro time cycle. They must come in ascending order as:
            [first_start, first_stop, second_start, second_stop, ...]. Therefore, 
            the number of elements in micro_time_gates must be an integer 
            multiple of 2. It is NOT currently possible to define distinct micro 
            time gates for different routing channels.
        
        Returns
        ----------
        channels_spec in nested tuple format - see docstring of check_channels_spec for details.
        
        
        '''
        
        # Input check
        if not isint(channels_indices) and not isiterable(channels_indices):
            raise ValueError('Invalid input for channels_indices: Must be int or list of int')

        elif isiterable(channels_indices) and not np.all([isint(element) for element in channels_indices]):
            raise ValueError('Invalid input for channels_indices: Must be int or list of int')

        if not isiterable(micro_time_gates):
            if micro_time_gates == None:
                # No micro time gating
                # We simply hand the channel index/channels indices into 
                # check_channels_spec() to construct a standard channels_spec
                channels_spec = FCS_Fixer.check_channels_spec(channels_indices)
            else:
                raise ValueError('Invalid input for micro_time_gates: Must be None, or an iterable of floats enumerating [first_start, first_stop, second_start, second_stop, ...]')

        elif isiterable(micro_time_gates):
            # Looks like the user was trying to specify a micro time gating...

            # Whatever type of iterable it was, we convert it to array to allow the following calculations
            micro_time_gates = np.array(micro_time_gates)

            if not (np.all([isfloat(element) for element in micro_time_gates]) and \
                    np.all(micro_time_gates >= 0.) and \
                    np.all(micro_time_gates <= 1.) and \
                    np.all(np.diff(micro_time_gates) >= 0.) and \
                    (len(micro_time_gates) > 0 and len(micro_time_gates) % 2 == 0)):
                # ... but did something wrong.
                raise ValueError('Invalid input for micro_time_gates. Must be None, or an iterable of floats enumerating [first_start, first_stop, second_start, second_stop, ...]')
            
            else:
                # ... which looks good.
                # Convert the micro time gate specifier(s) into something we can use
                micro_time_cutoffs = []
                micro_time_gates_to_use = []
                last_stop = 0.
                gate_counter = 0
                
                for i_gate in range(0, len(micro_time_gates), 2):
                    start = micro_time_gates[i_gate]
                    stop = micro_time_gates[i_gate+1]

                    if start > last_stop:
                        # There was a gap to skip, we need a new cutoff
                        micro_time_cutoffs.append(start)
                        micro_time_gates_to_use.append(gate_counter + 1)
                        gate_counter += 2

                    else:
                        # No gap to skip - we do not need to define a new cutoff
                        micro_time_gates_to_use.append(gate_counter)
                        gate_counter += 1

                    if stop < 1.:
                        # We have a new stop cutoff
                        micro_time_cutoffs.append(stop)
                        last_stop = stop
                        
                    else:
                        # We stop at 1, so there is no need to define a new cutoff
                        pass
                    
                if len(micro_time_cutoffs) == 0:
                    # Will happen if the user defined (0., 1.). In that case it's trivial after all.
                    channels_spec = FCS_Fixer.check_channels_spec(channels_indices)

                else:
                    # We constructed something interesting, time to piece it together
                    # channels_spec = (tuple(channels_indices,), (tuple(micro_time_cutoffs), tuple(micro_time_gates_to_use)))
                    channels_spec = ((channels_indices,), ((*micro_time_cutoffs,), (*micro_time_gates_to_use,)))
                    
                
        else:
            raise ValueError('Invalid input for micro_time_gates. Must be None, or an iterable of floats enumerating [first_start, first_stop, second_start, second_stop, ...]')

        return channels_spec
    
    
    
    @staticmethod
    def check_channels_spec(channels_spec):
        '''
        Run a quick check if channels_spec is usable. As this is arguably the 
        most complicated user input, I try to give specific feedback here.
        
        Also used to convert the different allowed channels_spec formats into 
        a single normalized format for internal processing.

        Parameters
        ----------
        channels_spec : 
            channels specifier tuple to be checked. can be one of three things:
                
                1. simple int for a single channel without micro time gating
                
                2. tuple, list, or np.array (1D) of int for a sum channel without micro time gating
                
                3. nested tuple for one or multiple channels with micro time gating. 
                       In this case, you need this structure:
                       ((tuple_of_int_specifying_one_or_multiple_channels), ((tuple_of_float_specifying_PIE_gate_edges), (tuple_of_int_specifying_which_gates_to_use)))
                       
        Returns the same channels_spec in format 3 (which can wrap formats 1 and 2 with dummy values) for standardized downstream processing.
        

        '''
        
        if not (isint(channels_spec) or isiterable(channels_spec)):
            # Outermost level is wrong already
            raise ValueError('''channels_spec can have one out of three structures: 
                                     1. int for a single channel without micro time gating
                                     2. simple tuple or list or np.array (1D) of int for a sum channel without micro time gating
                                     3. nested tuple like this: ((int_or_tuple_of_int_specifying_channels), ((tuple_of_float_specifying_PIE_gate_edges), (tuple_of_int_specifying_which_gates_to_use)))
                                     Got:
                                     ''' + str(channels_spec))

        elif isint(channels_spec):
            # It's an int, wrap and return
            return ((channels_spec,),((),(0,)))
        
        elif type(channels_spec) == list:
            # It could be a simple list...
            
            if isint(channels_spec[0]):
                # We really seem to be dealing with a simple list
                
                if np.any([not isint(element) for element in channels_spec]):
                    # At this point it is hard to say what the user wanted, so we give the full explanation.
                    raise ValueError('''channels_spec can have one out of three structures: 
                                     1. int for a single channel without micro time gating
                                     2. simple tuple or list or np.array (1D) of int for a sum channel without micro time gating
                                     3. nested tuple like this: ((int_or_tuple_of_int_specifying_channels), ((tuple_of_float_specifying_PIE_gate_edges), (tuple_of_int_specifying_which_gates_to_use)))
                                     Got:
                                     ''' + str(channels_spec))

                else:
                    # It's indeed a simple list of int and looks OK, wrap and return
                    return (tuple(channels_spec),((),(0,)))

        elif type(channels_spec) == np.array:
            # It could be a simple np.array...
            
            if isint(channels_spec[0])  and channels_spec.ndim == 1:
                # We really seem to be dealing with a simple np.array
                
                if np.any([not isint(element) for element in channels_spec]):
                    # At this point it is hard to say what the user wanted, so we give the full explanation.
                    raise ValueError('''channels_spec can have one out of three structures: 
                                     1. int for a single channel without micro time gating
                                     2. simple tuple or list or np.array (1D) of int for a sum channel without micro time gating
                                     3. nested tuple like this: ((int_or_tuple_of_int_specifying_channels), ((tuple_of_float_specifying_PIE_gate_edges), (tuple_of_int_specifying_which_gates_to_use)))
                                     Got:
                                     ''' + str(channels_spec))

                else:
                    # It's indeed a simple np.array of int and looks OK, wrap and return
                    return (tuple(channels_spec),((),(0,)))

        elif isiterable(channels_spec):
            # It is a tuple, tihs is the complicated situation. Let's look closer...
            
            if isint(channels_spec[0]):
                # We seem to be dealing with a simple tuple
                
                if np.any([not isint(element) for element in channels_spec]):
                    # At this point it is hard to say what the user wanted, so we give the full explanation.
                    raise ValueError('''channels_spec can have one out of three structures: 
                                     1. int for a single channel without micro time gating
                                     2. simple tuple or list or np.array (1D) of int for a sum channel without micro time gating
                                     3. nested tuple like this: ((int_or_tuple_of_int_specifying_channels), ((tuple_of_float_specifying_PIE_gate_edges), (tuple_of_int_specifying_which_gates_to_use)))
                                     Got:
                                     ''' + str(channels_spec))

                else:
                    # It's a simple tuple of int and looks OK, wrap and return
                    return (tuple(channels_spec),((),(0,)))
                
            if isfloat(channels_spec[0]):
                # Probably gave a float where an int was intended
                raise ValueError('''channels_spec[0] invalid. Found float where int was expected.''')

            if isiterable(channels_spec[0]):
                # This looks like a nested tuple, continue checks with that in mind
                
                # Check channel indices - should be tuple of int
                if np.any([isfloat(element) for element in channels_spec[0]]):
                    # Float instead of int
                    raise ValueError('''channels_spec[0] invalid. Found float where int was expected. 
                                     Got: 
                                     ''' + str(channels_spec))

                if np.any([not isint(element) for element in channels_spec[0]]):
                    # Something else...
                    raise ValueError('''channels_spec[0] invalid. channels_spec can have one out of three structures: 
                                     1. int for a single channel without micro time gating
                                     2. simple tuple or list or np.array (1D) of int for a sum channel without micro time gating
                                     3. nested tuple like this: ((int_or_tuple_of_int_specifying_channels), ((tuple_of_float_specifying_PIE_gate_edges), (tuple_of_int_specifying_which_gates_to_use)))
                                     Got:
                                     ''' + str(channels_spec))

                # Check second half
                if not isiterable(channels_spec[1]):
                    raise ValueError('''channels_spec[1] invalid. Found a tuple in tuple channels_spec[0], 
                                     which indicates that you tried to go with the nested tuple structure. 
                                     In that case, channels_spec[1] must be a nested tuple itself with structure: 
                                     ((tuple_of_float_specifying_PIE_gate_edges), (tuple_of_int_specifying_which_gates_to_use))
                                     Got: 
                                     ''' + str(channels_spec))

                else:
                    # channels_spec[1]it iterable, which makes sense, but are the inner tuples correct?
                    
                    # Check first element for micro time cutoffs.
                    if not isiterable(channels_spec[1][0]) or \
                        np.any([(not isfloat(element) or element < 0 or element > 1) for element in channels_spec[1][0]]):
                        raise ValueError('''channels_spec[1][0] invalid. Found a tuple in tuple channels_spec[0], 
                                         which indicates that you tried to go with the nested tuple structure. 
                                         In that case, channels_spec[1][0] must be a tuple of floats >= 0 and <= 1 
                                         defining the cutoff position relative to the full micro time dynamic range. 
                                         Got: 
                                         ''' + str(channels_spec))

                    # Check second element for selection of window(s).
                    if not isiterable(channels_spec[1][1]) or \
                        np.any([(not isint(element) or element > len(channels_spec[1][0])) for element in channels_spec[1][1]]):
                        raise ValueError('''channels_spec[1][1] invalid. Found a tuple in tuple channels_spec[0], 
                                         which indicates that you tried to go with the nested tuple structure. 
                                         In that case, channels_spec[1][1] must be a tuple of int specifying which
                                         gates defined by channels_spec[1][0] to use. Keep in mind that defining one 
                                         cutoff creates two windows with indices 0 and 1; defining two cutoffs 
                                         creates windows 0, 1, and 2, etc. 
                                         Got: 
                                         ''' + str(channels_spec))
                                         
                # If we arrived here, it should be a valid nested tuple structure, return normalizes structure
                return (tuple(channels_spec[0]), (tuple(channels_spec[1][0]), tuple(channels_spec[1][1])))
                                         
            
    @staticmethod
    def sort_start_stop(start_stop):
        '''
        Sort start_stop to merge consecutive segments.
        What this does: It looks at each segment in start_stop and looks if it 
        is seamlessly preceded or followed by another segment. Segment borders 
        between two segments are then dropped to reduce redundancy.

        The purpose is that if you segmented the trace somehow into a 
        start_stop structure and then classified the segments somehow, you can 
        crop away the segments to remove from the start_stop structure. This 
        function then merges the consecutive kept segments. Honestly mostly 
        for convenience and perhaps slight performance gain, does not do 
        anthing too important.

        Parameters
        ----------
        start_stop : 
            np.array (2D). Axis 0 is iteration over segments, axis 1 is 
            [first_photon_index_in_segment, first_photon_index_in_next_segment]

        Returns
        -------
        sorted start_stop : 
            Same as start_stop, but now merging segments between which nothing
            is discarded actually.

        '''

        pointer_start = 0
        pointer_stop = 0
        start_stop_sort = np.empty_like(start_stop)
        
        for segment_indx in range(start_stop.shape[0]):
            start = start_stop[segment_indx,0]
            stop = start_stop[segment_indx,1]
            
            if ((segment_indx == 0) or # First segment: We definitely need the start
                    (not start == start_stop[segment_indx-1,1])): # this good segement's start is not the stop of the proceding one
                # One of the criteria met: 
                start_stop_sort[pointer_start, 0] = start
                pointer_start += 1
                
            if ((segment_indx == start_stop.shape[0]-1) or # Last segment: We definitely need the end
                    (not stop == start_stop[segment_indx+1,0])): # this good segement's stop is not the start of the following one
                start_stop_sort[pointer_stop, 1] = stop
                pointer_stop += 1
                
        # After the loop, crop to remove the empty entries and return
        return start_stop_sort[:pointer_start,:]


    @staticmethod
    def get_flcs_filters(tcspc_y,
                         patterns):
        '''
        Use species-wise micro time patterns to calculate FLCS/fFCS unmixing
        filter weights.
        
        The actual filter calculation is slightly cryptic algebra...
        But it's basically a least-squares fit expressed via matrix inversion.
        If you want to understand more, check out the relevant literature, e.g.:
        Ghosh, ..., Enderlein Methods 2018, DOI: 10.1016/j.ymeth.2018.02.009


        Parameters
        ----------
        tcspc_y : 
            np.array (1D) with binned photon counts of TCSPC histogram.
        patterns : 
            np.array (2D) with (not necessarily normalized) micro time patterns 
            to unmix. Axis 0 is iteration over micro time bins, axis 1 is over
            patterns.


        Returns
        -------
        flcs_weights : 
            np.array (2D) of same shape as patterns. FLCS/fFCS filter weights
            for each pattern in each micro time bin.
        patterns_norm : 
            np.array (2D) of same shape as patterns. Essentially the same as 
            patterns, but each pattern is normalized.

        '''
                
        # Input check
        if not type(tcspc_y) == np.ndarray:
            raise ValueError('Invalid input for tcspc_y: Must be np.ndarray.')
            
        if not (type(patterns) == np.ndarray and patterns.ndim == 2 and patterns.shape[0] == tcspc_y.shape[0]):
            raise ValueError('Invalid input for patterns: Must be 2-dimensional np.ndarray, and of same length as tcspc_y along dimension 0.')
                
        # For fFCS filter calculation, we need the normalized patterns and a diagonal matrix derived from the total counts
        patterns_norm = np.ones_like(patterns, dtype = np.float64)
        
        for i_pattern in range(patterns.shape[1]):
            patterns_norm[:,i_pattern] = patterns[:,i_pattern] / patterns[:,i_pattern].sum()
            
        inv_tcspc_diag = np.diag(tcspc_y**(-1))

        # Actual filter weight calculation
        step_1 = np.matmul(inv_tcspc_diag, patterns_norm)
        step_2 = np.matmul(patterns_norm.T, step_1)
        step_3 = np.linalg.pinv(step_2)
        step_4 = np.matmul(patterns_norm, step_3)
        flcs_weights = np.matmul(inv_tcspc_diag, step_4)

        # Return
        return flcs_weights, patterns_norm


            
    #%% INIT METHOD 
    def __init__(self,
                 photon_data = None, # Dummy
                 out_path = None,
                 tau_min = 1E-6,
                 tau_max = 1.0,
                 sampling = 8,
                 cross_corr_symm = False,
                 correlation_method = 'default',
                 subtract_afterpulsing = False,
                 afterpulsing_params_path = '', # Default is dummy, must be specified if subtract_afterpulsing == True,
                 weights_ext = None,
                 write_results = True,
                 include_header = True,
                 write_log = True
                 ):
                 
        # Initialize some read-only properties that will be set later
        self._data_set = False
        self._parameters_set = False
        self._micro_time_corr = False
        
        # photon_data
        if type(photon_data) == tttrlib.TTTR:
            
            # Valid data: Write
            self._photon_data = photon_data   
            
            # Macro time tags and corrections
            self._macro_times = self._photon_data.macro_times 
            self._macro_times_correction_bursts = np.zeros_like(self._macro_times)
            self._macro_times_correction_mse_filter = np.zeros_like(self._macro_times)
            
            # Store some parameters for convenience
            self._micro_time_resolution = self._photon_data.header.micro_time_resolution
            self._macro_time_resolution = self._photon_data.header.macro_time_resolution
            self._acquisition_time = np.max(self._photon_data.macro_times) * self._macro_time_resolution
            self._n_total_photons = self._macro_times.shape[0]            
            self._n_micro_time_bins = self._photon_data.get_number_of_micro_time_channels()
            self._routing_channels = np.unique(self._photon_data.routing_channels)
            self._n_channels = np.max(self._routing_channels) + 1
            
            # Weights
            self._weights_burst_removal = np.ones_like(self._macro_times, dtype = np.bool8)
            self._weights_undrift = np.ones_like(self._macro_times, dtype = np.float16)
            self._weights_flcs_bg_corr = np.ones_like(self._macro_times, dtype = np.float16)
            self._weights_mse_filter = np.ones_like(self._macro_times, dtype = np.bool8)
            if weights_ext == None:
                self._weights_ext = np.ones_like(self._macro_times, dtype = np.float16)
            elif weights_ext.shape[0] == self._n_total_photons:
                self._weights_ext = weights_ext
            else:
                raise ValueError("weights_ext must be np.array with each single photon's weight!")
                
            # Register that data is loaded
            self._data_set = True
            
        else:
            self._photon_data = None
            self._macro_times = None
            self._macro_times_correction_bursts = None
            self._macro_times_correction_mse_filter = None
            self._micro_time_resolution = None
            self._macro_time_resolution = None
            self._acquisition_time = None
            self._n_total_photons = None
            self._n_micro_time_bins = None
            self._n_channels = None
            self._routing_channels = None
            self._weights_burst_removal = None
            self._weights_undrift = None
            self._weights_flcs_bg_corr = None
            self._weights_mse_filter = None
            self._weights_ext = None
        
        # out_path
        if out_path == None:
            # Empty: Create a default folder named based on a time tag
            self._out_path = os.path.join(os.getcwd(), datetime.datetime.now().strftime("%Y%m%d_%H%M") + 'FCS_Fixer_output') 
            
        elif os.path.isdir(out_path):
            # Exists already, we can work with that.
            self._out_path = out_path
            if not os.path.isdir(self._out_path):
                os.mkdir(self._out_path)
            
        else:
            # Does not yet exist - try making it
            try:
                os.mkdir(out_path)
                self._out_path = out_path
                
            except:
                # Failed
                raise ValueError("Invalid input encountered for out_path. Must be a valid directory name (existing or to be created) or None.")
                
        # tau_min and tau_max
        # First define defaults, then use child function to check input and possibly overwrite defaults
        self._tau_min = 1E3
        self._tau_max = 1E9
        self._tau_min, self._tau_max = self.check_tau_min_max(tau_min*1E9, tau_max*1E9)
            
        # sampling
        if type(sampling) == int and sampling > 0:
            self._sampling = sampling
            
        else:
            raise ValueError("Invalid input encountered for sampling. Must be positive int.")

        # cross_corr_symm
        if type(cross_corr_symm) == bool:
            self._cross_corr_symm = cross_corr_symm
            
        else:
            raise ValueError("Invalid input encountered for cross_corr_symm. Must be bool.")
        
        # correlation_method
        if correlation_method == 'default' or correlation_method == 'lamb':
            self._correlation_method = correlation_method
            
        else:
            raise ValueError("Invalid input encountered for correlation_method. Permitted are either 'default' or 'lamb'.")
        
        # subtract_afterpulsing
        if type(subtract_afterpulsing) == bool:
            self._subtract_afterpulsing = subtract_afterpulsing
            
        else:
            raise ValueError("Invalid input encountered for subtract_afterpulsing. Must be bool.")
            
        # afterpulsing_params_path
        if os.path.isfile(afterpulsing_params_path):
            self._afterpulsing_params_path = afterpulsing_params_path    
            self._afterpulsing_p = np.array([])
            self._afterpulsing_params = np.array([])    
            
        else:
            if self._subtract_afterpulsing:
                raise ValueError("Invalid input encountered for afterpulsing_params_path. If subtract_afterpulsing is set to True, this must be a valid file path string including file type suffix, referring to a correctly formatted calibration file. Got: " + str(afterpulsing_params_path))
            
            else:
                # Not used anyway, so we can ignore the invalid input and leave an empty dummy
                self._afterpulsing_params_path = ''
                self._afterpulsing_p = np.array([])
                self._afterpulsing_params = np.array([])            

        # write_results
        if type(write_results) == bool:
            self._write_results = write_results
            
        else:
            raise ValueError('Invalid input encountered for write_results. Must be bool.')

        # include_header
        if type(include_header) == bool:
            self._include_header = include_header
            
        else:
            raise ValueError('Invalid input encountered for include_header. Must be bool.')

        # out_file_counter
        # This is one that's always created at this point, no matter what input you give
        self._out_file_counter = 0

        # write_log
        if type(write_log) == bool:
            self._write_log = write_log
            if write_log:
                self._logfile_name = os.path.join(self._out_path, 'logfile.txt')   

                # Start by writing a bit about our dataset.
                # No need to log all the other parameters, these are logged whereever they are used.
                log_message = f'''Writing output into {self._out_path}.
                '''
                if self._data_set:
                    photons_in_channels_str = ', '.join([str(channel) + ': ' + str(self._photon_data.get_selection_by_channel([channel]).shape[0]) for channel in self._routing_channels])
                    log_message += f'''
                    Overview over loaded TTTR data:
                    {self._n_total_photons} photons in total
                    Channels in use and how many photons are in each of them:
                    '''+ photons_in_channels_str + f'''
                    {self._acquisition_time*1E-9} s acquisition time
                    {self._n_micro_time_bins} micro time bins
                    {self._macro_time_resolution} ns macro time resolution
                    {self._micro_time_resolution} ns micro time resolution
                    '''
                else:
                    log_message += '''
                    No data loaded yet.'''
                     
                self.write_to_logfile(log_header = '''Creation:''',
                                      log_message = log_message)
                
            else:
                self._logfile_name = ''
                
        else:
            raise ValueError('Invalid input encountered for write_log. Must be bool.')



    #%% READ-ONLY PROPERTIES
    @property
    def data_set(self):
        '''Reports whether a TTTR data object has been loaded.'''
        return self._data_set
    
    @property
    def parameters_set(self): 
        '''Reports whether all parameters including derived ones based on both data and args are complete.'''
        return self._parameters_set
    
    @property
    def logfile_name(self): 
        '''Path of logfile.'''
        return self._logfile_name

    @property
    def micro_time_corr(self): 
        '''Indicates whether "micro-time", i.e. sub-nanosecond, time resolution is used in correlation.
        Software adapts this automatically to the queried time scales.'''
        return self._micro_time_corr  
    
    @property
    def sd_Wohland_min_window_auto(self): 
        '''Indicates whether time window for Wohland method SD calculation is tuned automatically. 
        Set through input in sd_min_window.'''
        return self._sd_Wohland_min_window_auto
    
    @property
    def afterpulsing_params(self): 
        '''Returns afterpulsing parameters, with time constants (cols 1 and 3) in seconds and amplitudes (cols 0 and 2) in Hz.'''
        afterpulsing_params_ret = self._afterpulsing_params.copy()
        afterpulsing_params_ret[:, [0,2]] *= 1E9 # Convert amplitudes stored in GHz to Hz for reporting
        afterpulsing_params_ret[:, [1,3]] *= 1E-9 # Convert time constants stored in ns to seconds for reporting
        return afterpulsing_params_ret
    
    @property
    def afterpulsing_p(self):
        '''Returns overall afterpulsing probabilities for all detectors in calibration.'''
        return self._afterpulsing_p
    
    @property
    def n_casc(self): 
        '''Indicates how many cascades of lag times to calculate in cc calculation. 
        Automatically set based on other CC sampling parameters.'''
        return self._n_casc
    
    @property
    def acquisition_time(self):
        '''Returns the acquisition time  in seconds, estimated through the time tag of the latest photon.'''
        return self._acquisition_time * 1E-9 # Convert time stored in ns to seconds for reporting
    
    @property
    def total_photons(self):
        '''Returns the total number of photons in the TTTR object.'''
        return self._n_total_photons
    
    @property
    def n_micro_time_bins(self):
        '''Returns the total number of micro time bins in the TTTR object.'''
        return self._n_micro_time_bins
    
    @property
    def n_channels(self):
        '''Returns the total number of channels (spectral/polarization/...) in the TTTR object.'''
        return self._n_channels

    @property
    def routing_channels(self):
        '''Returns an array listing the channels (spectral/polarization/...) in the TTTR object.'''
        return self._routing_channels

    @property
    def micro_time_resolution(self):
        '''Returns the TCSPC time resolution of the TTTR object in seconds.'''
        return self._micro_time_resolution * 1E-9
    
    @property
    def macro_time_resolution(self):
        '''Returns the macro time resolution of the TTTR object in seconds.'''
        return self._macro_time_resolution * 1E-9
    
    @property
    def micro_times(self):
        '''Returns micro times of photons in seconds.'''
        return self.photon_data.micro_times * self._micro_time_resolution * 1E-9
    
    @property
    def weights_burst_removal(self):
        '''Returns photon-by-photon indicator if this photon is to be used based 
        on burst removal or not. It is a logical inverse of "Is this photon part of a burst?"''' 
        return self._weights_burst_removal
    
    @property
    def weights_anomalous_segments(self):
        '''Returns photon-by-photon indicator if this photon is to be used based on anomalous segment removal or not.''' 
        return self._weights_mse_filter
    
    @property
    def weights_burst_undrift(self):
        '''Returns photon-by-photon weight for bleaching/drift correction.''' 
        return self._weights_undrift

    @property
    def weights_flcs_bg_corr(self):
        '''Returns photon-by-photon weight for FLCS-based background correction.''' 
        return self._weights_flcs_bg_corr
    
    @property
    def photon_weights(self):
        '''Returns the overall weight of each photon, considering all weights contributions.'''
        return self._weights_burst_removal * self._weights_ext * self._weights_mse_filter * self._weights_undrift * self._weights_flcs_bg_corr
        
    @property
    def macro_times_burst_corrected(self):
        '''Returns the macro_time tags in seconds of photons with correction of burst times applied.'''
        return (self._macro_times - self._macro_times_correction_bursts) * self._macro_time_resolution * 1E-9

    @property
    def out_file_counter(self):
        '''Returns a running number of output files creates.'''
        return self._out_file_counter
        
    #%% READ-WRITE PROPERTIES
    
    # photon_data: Actual data to be processed
    @property
    def photon_data(self):
        '''
        tttrlib.TTTR object that is to be processed. Can also be temporarily 
        set as a dummy value, leaving the associated attributes with None values,
        but you won't actually be able to do anything with that.
        '''
        return self._photon_data
    
    @photon_data.setter
    def photon_data(self, new_photon_data):
        # photon_data
        if type(new_photon_data) == tttrlib.TTTR:
            
            # Valid data: Write
            self._photon_data = new_photon_data   
            
            # Macro time tags and corrections
            self._macro_times = self._photon_data.macro_times 
            self._macro_times_correction_bursts = np.zeros_like(self._macro_times)
            self._macro_times_correction_mse_filter = np.zeros_like(self._macro_times)
            
            # Store some parameters for convenience
            self._micro_time_resolution = self._photon_data.header.micro_time_resolution
            self._macro_time_resolution = self._photon_data.header.macro_time_resolution
            self._acquisition_time = np.max(self._photon_data.macro_times) * self._macro_time_resolution
            self._n_total_photons = self._macro_times.shape[0]            
            self._n_micro_time_bins = self._photon_data.get_number_of_micro_time_channels()
            self._routing_channels = np.unique(self._photon_data.routing_channels)
            self._n_channels = np.max(self._routing_channels) + 1
            
            self._weights_burst_removal = np.ones_like(self._macro_times, dtype = np.bool8)
            self._weights_undrift = np.ones_like(self._macro_times, dtype = np.float16)
            self._weights_flcs_bg_corr = np.ones_like(self._macro_times, dtype = np.float16)
            self._weights_mse_filter = np.ones_like(self._macro_times, dtype = np.bool8)
            self._weights_ext = np.ones_like(self._macro_times, dtype = np.float16)

            # Register that data is loaded
            self._data_set = True
            self._parameters_set = False
                        
        else:
            self._photon_data = None
            self._macro_times = None
            self._macro_times_correction_bursts = None
            self._macro_times_correction_mse_filter = None
            self._micro_time_resolution = None
            self._macro_time_resolution = None
            self._acquisition_time = None
            self._n_total_photons = None
            self._n_micro_time_bins = None
            self._n_channels = None
            self._routing_channels = None
            self._weights_burst_removal = None
            self._weights_undrift = None
            self._weights_flcs_bg_corr = None
            self._weights_mse_filter = None
            self._weights_ext = None
            
            self._data_set = False
            self._parameters_set = False

        if self._write_log:
            # Log a bit about our dataset.
            if self._data_set:
                photons_in_channels_str = ', '.join([str(channel) + ': ' + str(self._photon_data.get_selection_by_channel([channel]).shape[0]) for channel in self._routing_channels])
                log_message = f'''Overview over loaded TTTR data:
                {self._n_total_photons} photons in total
                Channels in use and how many photons are in each of them:
                '''+ photons_in_channels_str + f'''
                {self._acquisition_time*1E-9} s acquisition time
                {self._n_micro_time_bins} micro time bins
                {self._macro_time_resolution} ns macro time resolution
                {self._micro_time_resolution} ns micro time resolution
                '''
            else:
                log_message = '''No data loaded.'''
                 
            self.write_to_logfile(log_header = '''Overwrote loaded dataset.''',
                                  log_message = log_message)

    @photon_data.getter
    def photon_data(self):
        return self._photon_data
    
    @photon_data.deleter
    def photon_data(self):
        # Remove data and all data-dependent properties.
        self._photon_data = None
        self._macro_times = None
        self._macro_times_correction_bursts = None
        self._macro_times_correction_mse_filter = None
        self._micro_time_resolution = None
        self._macro_time_resolution = None
        self._acquisition_time = None
        self._n_total_photons = None
        self._n_micro_time_bins = None
        self._n_channels = None

        self._weights_burst_removal = None
        self._weights_mse_filter = None
        self._weights_undrift = None
        self._weights_ext = None
        
        self._data_set = False
        self._parameters_set = False

        if self._write_log:
            self.write_to_logfile(log_header = '''Overwrote loaded dataset.''',
                                  log_message = '''No data loaded.''')
            
            
    # macro_times: (potentially shifted) macro times of photons.
    @property
    def macro_times(self):
        '''
        Macro time tags of loaded photons in seconds.
        
        You can set user-defined macro_times (np.array of length equal length of 
        TTTR object). This setter is offered for custom macro time tag 
        manipulation, but there are no sanity checks in this class for your 
        input here, and the code is honestly not built with this setter being 
        used in mind. Use entirely at your own risk. Input is expected in time 
        units of seconds.
        
        Deleting the macro_times attribute is also possible, which resets the 
        values to the raw data in the loaded TTTR object.
        '''
        return self._macro_times * self._macro_time_resolution * 1E-9 # Return in seconds
    
    @macro_times.setter
    def macro_times(self, new_macro_times):
        if self._data_set:
            if type(new_macro_times) == np.ndarray and new_macro_times.shape[0] == self._macro_times.shape[0]:
                new_macro_times_convert = new_macro_times / self._macro_time_resolution * 1E9 # Write in macro_time_resolution bins
                self._macro_times = np.uint64(new_macro_times_convert)
                
            else:
                raise ValueError('Invalid input for macro_times: Must be np.array of same length as loaded TTTR object.')
                
        else:
            raise RuntimeError('Before loading macro_times, load a TTTR object.') 
            
    @macro_times.getter
    def macro_times(self):
        return self._macro_times * self._macro_time_resolution * 1E-9 # Return in seconds
    
    @macro_times.deleter
    def macro_times(self):
        if self._data_set:
            self._macro_times = self.photon_data.macro_times
        else:
            self._macro_times = None
        
    # out_path
    @property
    def out_path(self):
        '''
        Directory in which to save output files created by this object.
        '''
        return self._out_path
    
    @out_path.setter
    def out_path(self, new_out_path):
        if new_out_path == None:
            # Empty: Create a default folder named based on a time tag
            self._out_path = os.path.join(os.getcwd(), datetime.datetime.now().strftime("%Y%m%d_%H%M") + 'CFcleaner_output') 
            if not os.path.isdir(self._out_path):
                os.mkdir(self._out_path)
                
        elif os.path.isdir(new_out_path):
            # Exists already, we can work with that.
            self._out_path = new_out_path
            
        else:
            # Does not yet exist - try making it
            try:
                os.mkdir(new_out_path)
                self._out_path = new_out_path
                
            except:
                # Failed
                raise ValueError("Invalid input encountered for out_path. Must be a valid directory name (existing or to be created) or None.")
                
    @out_path.getter
    def out_path(self):
        return self._out_path
    
    @out_path.deleter
    def out_path(self):
        # Default directory
        self._out_path = os.path.join(os.getcwd(), datetime.datetime.now().strftime("%Y%m%d_%H%M") + 'CFcleaner_output') 
        if not os.path.isdir(self._out_path):
            os.mkdir(self._out_path)


    # tau_min
    @property
    def tau_min(self):
        '''Minimum correlation lag time to analyze in seconds.'''
        return self._tau_min / 1E-9
    
    @tau_min.setter
    def tau_min(self, new_tau_min):
        # First define default, then use child function to check input and possibly overwrite default
        self._tau_min = 1E3
        self._tau_min, _ = self.check_tau_min_max(tau_min = new_tau_min * 1E9)
            
    @tau_min.getter
    def tau_min(self): # Return in seconds
        return self._tau_min / 1E-9


    # tau_max
    @property
    def tau_max(self):
        '''Maximum correlation lag time to analyze in seconds.'''
        return self._tau_max
    
    @tau_max.setter
    def tau_max(self, new_tau_max):
        # First define default, then use child function to check input and possibly overwrite default
        self._tau_max = 1E9
        _, self._tau_max = self.check_tau_min_max(tau_max = new_tau_max * 1E9)
            
    @tau_max.getter
    def tau_max(self): # Return in seconds
        return self._tau_max / 1E-9


    # sampling
    @property
    def sampling(self):
        '''Density of correlation function sampling.'''
        return self._sampling
    
    @sampling.setter
    def sampling(self, new_sampling):
        if type(new_sampling) == int and new_sampling < 0:
            self._sampling = new_sampling
            self._parameters_set = False
            
        else:
            raise ValueError("Invalid input encountered for sampling. Must be positive int.")
            
    @sampling.getter
    def sampling(self):
        return self._sampling


    # cross_corr_symm
    @property
    def cross_corr_symm(self):
        '''
        When calculating cross-correlation between two nonredundant 
        channels, do you assume time symmetry (thermodynamic equilibrium)? 
        If yes, cross-correlation functions will be averaged in forward-time and
        backward-time. Doing so increases SNR, but must be physically 
        justified (which is often the case though).
        '''
        return self._cross_corr_symm
    
    @cross_corr_symm.setter
    def cross_corr_symm(self, new_cross_corr_symm):
        if type(new_cross_corr_symm) == bool:
            self._cross_corr_symm = new_cross_corr_symm
            self._parameters_set = False
            
        else:
            raise ValueError("Invalid input encountered for cross_corr_symm. Must be bool.")
            
    @cross_corr_symm.getter
    def cross_corr_symm(self):
        return self._cross_corr_symm
        
    
    # correlation_method
    @property
    def correlation_method(self):
        '''
        Which code to use in the backend for the actual correlation function calculation?
        String that must be either 'default' or 'lamb'
        '''
        return self._correlation_method
    
    @correlation_method.setter
    def correlation_method(self, new_correlation_method):
        if new_correlation_method == 'default' or new_correlation_method == 'lamb':
            self._correlation_method = new_correlation_method
            
        else:
            raise ValueError("Invalid input encountered for correlation_method. Permitted are either 'default' or 'lamb'.")
            
    @correlation_method.getter
    def correlation_method(self):
        return self._correlation_method


    # subtract_afterpulsing
    @property
    def subtract_afterpulsing(self):
        '''
        Do you want to perform calibrated subtraction of afterpulses? 
        Should be left off if you plan to use FLCS.
        '''
        return self._subtract_afterpulsing
    
    @subtract_afterpulsing.setter
    def subtract_afterpulsing(self, new_subtract_afterpulsing):
        if type(new_subtract_afterpulsing) == bool:
            self._subtract_afterpulsing = new_subtract_afterpulsing
            self._parameters_set = False
            if self._subtract_afterpulsing and len(self._afterpulsing_params_path) == 0:
                # self._afterpulsing_params_path is needed now, but not defined!
                raise Warning('subtract_afterpulsing has been set to True, but afterpulsing_params_path is missing. This will lead to an error if left uncompensated!')
                
        else:
            raise ValueError("Invalid input encountered for subtract_afterpulsing. Must be bool.")
            
    @subtract_afterpulsing.getter
    def subtract_afterpulsing(self):
        return self._subtract_afterpulsing


    # afterpulsing_params_path
    @property
    def afterpulsing_params_path(self):
        '''Path (directory and file name) to calibration csv file with parameters for calibrated afterpulsing subtraction.'''
        return self._afterpulsing_params_path
    
    @afterpulsing_params_path.setter
    def afterpulsing_params_path(self, new_afterpulsing_params_path):
        if os.path.isfile(new_afterpulsing_params_path):
            self._afterpulsing_params_path = new_afterpulsing_params_path            
            self._afterpulsing_p = np.array([])
            self._afterpulsing_params = np.array([])
            self._parameters_set = False
            
        else:
            if self._subtract_afterpulsing:
                raise ValueError("Invalid input encountered for afterpulsing_params_path. If subtract_afterpulsing is set to True, this must be a valid file path string including file type suffix, referring to a correctly formatted calibration file.")
            
            else:
                # Not used anyway, so we can ignore the invalid input and leave empty dummies
                self._afterpulsing_params_path = ''            
                self._afterpulsing_p = np.array([])
                self._afterpulsing_params = np.array([])
                self._parameters_set = False
                
    @afterpulsing_params_path.getter
    def afterpulsing_params_path(self):
        return self._afterpulsing_params_path


    # write_log
    @property
    def write_log(self):
        '''Flag whether a logfile is to be kept or not.'''
        return self._write_log
    
    @write_log.setter
    def write_log(self, new_write_log):
        if type(new_write_log) == bool:
                        
            if new_write_log and not self._write_log:
                # We start writing the log
                self._write_log = new_write_log
                
                self._logfile_name = os.path.join(self._out_path, 'logfile.txt')
                
                # Start by writing a bit about our dataset.
                # No need to log all the other parameters, these are logged whereever they are used.
                log_message = f'''Writing output into {self._out_path}.
                '''
                
                if self._data_set:
                    photons_in_channels_str = ', '.join([str(channel) + ': ' + str(self._photon_data.get_selection_by_channel([channel]).shape[0]) for channel in self._routing_channels])
                    log_message += f'''
                    Overview over loaded TTTR data:
                    {self._n_total_photons} photons in total
                    Channels in use and how many photons are in each of them:
                    '''+ photons_in_channels_str + f'''
                    {self._acquisition_time*1E-9} s acquisition time
                    {self._n_micro_time_bins} micro time bins
                    {self._macro_time_resolution} ns macro time resolution
                    {self._micro_time_resolution} ns micro time resolution
                    '''
                    
                else:
                    log_message += '''
                    No data loaded yet.'''
                     
                self.write_to_logfile(log_header = '''Creation:''',
                                      log_message = log_message)  
                
            elif self._write_log and not new_write_log:
                # We stop writing the log
                self._write_log = new_write_log
                
                self.write_to_logfile(log_header = '''Stopped logging.''',
                                      log_message = log_message)
                
            else:
                # Nothing changed actually, but just in case I overlooked some obscure logical case, we explicitly overwrite anyway
                self._write_log = new_write_log

        else:
            raise ValueError('Invalid input encountered for write_log. Must be bool.')
            
    @write_log.getter
    def write_log(self):
        return self._write_log


    # weights_ext
    @property
    def weights_ext(self):
        '''This can be used to define external photon weighting functions, i.e., FLCS/fFCS filters.'''
        return self._weights_ext
    
    @weights_ext.setter
    def weights_ext(self, new_weights_ext):
        if type(new_weights_ext) == np.ndarray and new_weights_ext.shape[0] == self._macro_times.shape[0]:
            self._weights_ext = new_weights_ext
            
        else:
            raise ValueError("Invalid input encountered for weights_ext. Must be np.array of the same length as the loaded TTTR object.")
            
    @weights_ext.getter
    def weights_ext(self):
        return self._weights_ext
    
    @weights_ext.deleter
    def weights_ext(self):
        if self._data_set: 
            # If no data is set, there is nothing to do at this point.
            self._weights_ext = np.ones_like(self._macro_times)    


    # write_results
    @property
    def write_results(self):
        '''Whether to automatically write output .csv and .png files from suitable functions.'''
        return self._write_results
    
    @write_results.setter
    def write_results(self, new_write_results):
        if type(new_write_results) == bool:
            self._write_results = new_write_results
            
        else:
            raise ValueError("Invalid input encountered for write_results. Must be bool.")
            
    @write_results.getter
    def write_results(self):
        return self._write_results
    

    # include_header
    @property
    def include_header(self):
        '''Relevant only if write_results == True. Whether to write the results table with header row for better human readability, or without header for easier machine readability.'''
        return self._include_header
    
    @include_header.setter
    def include_header(self, new_include_header):
        if type(new_include_header) == bool:
            self._include_header = new_include_header
            
        else:
            raise ValueError("Invalid input encountered for include_header. Must be bool.")
            
    @include_header.getter
    def include_header(self):
        return self._include_header

        


    #%% Instance methods - GENERAL
    
    def set_correlation_time_params(self,
                                    suppress_logging = False):
        '''
        Use various properties to get parameters relating to the lag times to use in the correlation operation.


        Parameters
        ----------
        suppress_logging : Bool, optional
            If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the Class instance has been 
            set to create a log file. The default is False.

        Returns
        -------
        None.

        '''
        # Check what time resolution is needed
        if self._tau_min < 20 * self._macro_time_resolution:
            # High time resolution needed
            self._micro_time_corr = True
            resolution = self._micro_time_resolution
            
        else:
            # Moderate time resolution sufficient
            self._micro_time_corr = False
            resolution = self._macro_time_resolution
        
        # Coarse (over-)estimation of how many correlation cascades are needed
        lag_point = 1
        lag_time = resolution
        while lag_time <= 10. * self._tau_max: 
            lag_point += 1
            lag_time += 2**(np.floor((lag_point-1)/self._sampling)) * resolution
            
        self._n_casc = np.ceil(lag_point/self._sampling)
        
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            if self._subtract_afterpulsing:
                self.write_to_logfile(log_header = '''set_correlation_time_params: Setting correlator settings''',
                                      log_message = f'''Parameters set:
                                          micro_time_corr (Use of micro time information in correlation): {self._micro_time_corr}
                                          sampling (number of samples per correlator cascade): {self._sampling}
                                          n_casc (number of correlation cascades): {self._n_casc}
                                          ''')

        
    def set_afterpulsing_params(self,
                                suppress_logging = False):
        '''
        Load the afterpulsing calibration file and store the parameters in properties.
        The file is expected to be a csv file with one header line and 
        afterpulsing properties in a biexponential model for each channel, 
        line-wise like so (example for a two-channel setup):
            
        A1,tau1,A2,tau2
        137.47,6.77435e-06,7346.93,3.13849e-07
        306.463,7.43791e-06,4105.75,3.10915e-07
        
        All values floats, amplitudes in Hz, time constants in seconds.


        Parameters
        ----------
        suppress_logging : Bool, optional
            If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the Class instance has been 
            set to create a log file. The default is False.

        '''
        
        if self._subtract_afterpulsing:
            afterpulsing_params = np.genfromtxt(self._afterpulsing_params_path, delimiter=',', skip_header=1)
            afterpulsing_params[:, [0,2]] *= 1E-9 # Convert amplitudes from Hz to GHz for storage
            afterpulsing_params[:, [1,3]] *= 1E9 # Convert time constants from seconds to ns for storage
            self._afterpulsing_params = afterpulsing_params
            
            # For convenience, also predict and store afterpulsing probability
            self._afterpulsing_p = afterpulsing_params[:,0]*afterpulsing_params[:,1]+afterpulsing_params[:,2]*afterpulsing_params[:,3]
        
        else:
            self._afterpulsing_params = np.zeros((self._n_channels, 4))
            self._afterpulsing_params[:,[1,3]] = 1.
            self._afterpulsing_p = np.zeros((self._n_channels,)) # Dummy
            
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            if self._subtract_afterpulsing:
                self.write_to_logfile(log_header = '''set_afterpulsing_params: Setting afterpulsing calibration parameters''',
                                      log_message = f'''loaded afterpulsing calibration file {self._afterpulsing_params_path}
                                      Parameters:
                                          A1[GHz], tau1[ns], A2[GHz], tau2[ns]:
                                          {self._afterpulsing_params}
                                      Channel-wise afterpulsing probabilities according to calibration:
                                          {self._afterpulsing_p}
                                          ''')
                                          
            else:
                self.write_to_logfile(log_header = '''set_afterpulsing_params: Setting afterpulsing calibration parameters''',
                                      log_message = f'''Created dummy afteroulsing calibration parameters
                                        Parameters:
                                        A1[GHz], tau1[ns], A2[GHz], tau2[ns]:
                                        {self._afterpulsing_params}
                                        ''')

                
    def update_params(self):
        '''
        Use parameters set in init() method or set/updated later and data to 
        define some derived parameters.
        '''
        if not self._data_set: # Data loaded?
            raise RuntimeError('Cannot proceed as TTTR data has not been loaded.')

        if not self._parameters_set: # Derived parameters set?
            
            # Set correlation lag time settings
            self.set_correlation_time_params()
            
            # Load detector parameters
            self.set_afterpulsing_params()
            
            # Done
            self._parameters_set = True
            
        else: # parameters have been set already, nothing to update
            pass
    
    
    def check_tau_min_max(self, 
                          tau_min = None, 
                          tau_max = None):
        '''
        A little bit of logic that is needed at different places. Basically, 
        take the input for tau_min and tau_max, check if it is something we can work with,
        and if empty input is found, set the default from instance parameters.
        
        Arguments are optional to give the option of checking only one of the two.
        
        INPUTS:
            tau_min: 
                 Minimum lag time to correlate. float in ns.
                 If empty (empty np.array, list, or tuple, or None), it will 
                 be replaced by the value set as class instance parameter.
            tau_max:
                 Maximum lag time to correlate. float in ns. 
                 If empty (empty np.array, list, or tuple, or None), it will 
                 be replaced by the value set as class instance parameter.
        OUTPUTS:
            tau_min, tau_max: 
                Same as above, but now definitely float that we can work with.
        
        '''
        
        # tau_min
        if tau_min == None:
            # Set default
            tau_min = self._tau_min
            
        elif type(tau_min) in [np.ndarray, tuple, list]:
            if len(tau_min) == 0: 
                # Empty iterable: Set default
                tau_min = self._tau_min
                
            elif type(tau_min[0]) in [float, np.float32, np.float64] and tau_min > 0 and len(tau_min == 1):
                # Iterable of size 1 with float > 0 - works
                tau_min = tau_min[0]
            
            else:
                # Something else - NO.
                raise ValueError("Invalid input encountered for tau_min. Can be float, empty array (use default), or None (use default).")
                
        elif type(tau_min) in [float, np.float32, np.float64] and tau_min > 0:
            # Single float > 0 - Works
            pass
        
        else:
            # Something else - NO.
            raise ValueError("Invalid input encountered for tau_min. Can be float, empty array (use default), or None (use default).")

        # tau_max
        if tau_max == None:
            # Set default
            tau_max = self._tau_max
            
        elif type(tau_max) in [np.ndarray, tuple, list]:
            if len(tau_max) == 0: 
                # Empty iterable: Set default
                tau_max = self._tau_max
                
            elif type(tau_max[0]) in [float, np.float16, np.float32, np.float64] and tau_max[0] > tau_min and len(tau_min == 1):
                # Iterable of size 1 with float > tau_min - works
                tau_max = tau_max[0]
            
            else:
                # Something else - NO.
                raise ValueError("Invalid input encountered for tau_max. Can be float > tau_min, empty array (use default), or None (use default).")
                
        elif type(tau_max) in [float, np.float16, np.float32, np.float64] and tau_max > tau_min:
            # Single float > tau_min - Works
            pass
        
        else:
            # Something else - NO.
            raise ValueError("Invalid input encountered for tau_max. Can be float > tau_min, empty array (use default), or None (use default).")

        return tau_min, tau_max
    
    
    def write_to_logfile(self,
                         log_header = '',
                         log_message = '',
                         calling_function = ''):
        '''
        Write string-formatted information into the logfile. The input is two 
        strings to give the option of making things tidy through separation of 
        header and text body, plus a third string that acts as a header 
        extension, which is meant specifically for allowing the user to track
        in the log which call of one function to another did what.
        
        Further, the log entry will auto-create a time stamp, and a counter of 
        the number of outut files auto-created so far, which again helps track
        in the log which function calls are related to each other (and to which
        output - note that this number is equal to the index of the next output 
        file to be auto-created).

        Parameters
        ----------
        log_header : string, optional
            First message to write, which is presented rather as header. The default is ''.
        log_message : string, optional
            Second message to write, which is presented as text body. The default is ''.
        calling_function : string, optional
            This is another handle specifically meant for logging which function 
            had called the function that is being logged. To make the code stack 
            more understandable in the log.
            
        Returns
        -------
        None.

        '''
        

        # Input check
        if type(log_header) == str and type(log_message) == str and type(calling_function) == str:
            
            # If we have info on the calling function, append the header accordingly
            if len(calling_function) > 0:
                log_header += '''
                (Called by ''' + calling_function + ')'
            
            # Get full log entry, doing two things in one step:
            # - Concatenate string parts
            # - Add time stamp
            logfile_entry_full = '''


            ---------------------------------------------
            ---------------------------------------------
            '''+log_header+f'''
            {datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}
            {self._out_file_counter} output files generated so far.
            
            '''+log_message+'''
            '''
            
            # Also, remove tabs (4-blank-space groups) to make format in log 
            # more regular without making the code messy
            logfile_entry_notabs = logfile_entry_full.replace('    ', '')
                          
            # Write to file
            if os.path.isfile(self._logfile_name):
                # File exists, we append
                with open(self._logfile_name, 'a') as logfile:
                    logfile.write(logfile_entry_notabs)
                                  
            else:
                # File does not exist yet, we create
                with open(self._logfile_name, 'w') as logfile:
                    logfile.write(logfile_entry_notabs)
                                  
        else:
            raise ValueError('Invalid input in write_to_logfile(). Input must be string.')
    
    
    def get_channel_combinations(self,
                                 min_photons = 0):
        '''
        Return a list of 2-tuples of channels_spec tuples that specify all 
        channel pairs for "simple" (no micro time gating) correlation operations
        based on what channels in the data contain any photons.

        Convenience function if you want to run all-against-all correlations of 
        the channels in your data without requiring you to think about micro 
        time gating.
        
        Parameters
        -------
        min_photons :
            Int. If specified and >0, this variable specifies the minimum 
            number of photons that the channel must record to be included in 
            analysis. Helpful for data from setups that tend to create a very 
            small number of spurious pseudo-photons in channels that are not 
            even in use.
        
        Returns
        -------
        list_of_channels_spec_pairs : 
            List of 2-tuples of channels_spec tuples.

        '''
        
        # Input check, figuring out which channels to use

        if isint(min_photons) and min_photons > 0:
            routing_channels_use = []
            for channel in self._routing_channels: 
                if self._photon_data.get_selection_by_channel([channel]).shape[0] >= min_photons:
                    routing_channels_use.append(channel)
                    
        elif min_photons == 0:
            routing_channels_use = self._routing_channels
            
        else:
            raise ValueError('Invalid input for min_photons in get_channel_combinations(). Must be int >= 0')
        
        
        # Enumerate channel combinations without creating duplicates
        # The exact procedure depends on whether or not we assume time symmetry
        list_of_channels_spec_pairs = []
        
        if self._cross_corr_symm:
            # in this case, a single list-comprehension-in-for-loop structure is enough
            for first_channel in routing_channels_use:
                [list_of_channels_spec_pairs.append((((first_channel,), ((), (0,))), ((second_channel,), ((), (0,))))) for second_channel in routing_channels_use if second_channel >= first_channel]

        else:
            # Here we use a list comprehension for autocorrelation, and a list-comprehension-in-for-loop structure for cross-correlation
            [list_of_channels_spec_pairs.append((((channel,), ((), (0,))), ((channel,), ((), (0,))))) for channel in self._routing_channels]
            
            for first_channel in routing_channels_use:
                [list_of_channels_spec_pairs.append((((first_channel,), ((), (0,))), ((second_channel,), ((), (0,))))) for second_channel in routing_channels_use if second_channel != first_channel]

        return list_of_channels_spec_pairs
    
        
    #%% Instance methods - CORRELATION AND TIME TRACES
    def simple_correlation(self, 
                            macro_times_ch1, 
                            macro_times_ch2, 
                            tau_min = None, 
                            tau_max = None): 
        ''' 
        Perform a very basic correlation function calculation, without any 
        filters or such.
        
        Honestly, this function has relatively little value in actual practical
        use given what this class is built to do. Think of it as this class' 
        equivalent of print('Hello World').
        
        INPUTS:
            macro_times_ch1, macro_times_ch2: 
                macro time tags (in units of macro_time_resolution) that form 
                the photons in the two channels to correlate.
            tau_min, tau_max:
                OPTIONAL specifiers for minimum and maximum lag time to correlate. 
                float in ns. If empty or not specified at all, will be replaced 
                by class instance defaults.
                
        OUTPUTS:
            lag_times: 
                correlation lag times (x axis) in ns
            cc: 
                Correlation function (with +1 offset)
        '''
        
        # Make sure that correlation parameters are complete
        if not self._parameters_set:
            self.update_params()
        
        # Set up correlator object and set its parameters
        corr = tttrlib.Correlator()
        corr.n_bins = self._sampling
        corr.n_casc = int(self._n_casc) # tttrlib.Correlator.n_casc setter is a bit stubborn and really insists on this int() conversion
        corr.method = self._correlation_method
        corr.make_fine = False
        
        # Run input check for limits of correlation times to use.
        tau_min, tau_max = self.check_tau_min_max(tau_min, tau_max)
        
        # Perform correlation 
        w1 = np.ones_like(macro_times_ch1, dtype = float) # Dummy
        w2 = np.ones_like(macro_times_ch2, dtype = float) # Dummy
        corr.set_events(macro_times_ch1, w1, macro_times_ch2, w2)
        cc = corr.get_corr_normalized()
        
        # Crop to queried lag time range, and return
        lag_times = corr.get_x_axis_normalized() * self._macro_time_resolution
        keep = np.logical_and(lag_times >= tau_min, lag_times <= tau_max)

        return lag_times[keep], cc[keep]
        
    
        
    def select_photons(self,
                       channels_spec, 
                       ext_indices = np.array([]),
                       use_ext_weights = False, 
                       use_drift_correction = False,
                       use_flcs_bg_corr = False,
                       use_burst_removal = False,
                       use_mse_filter = False,
                       suppress_logging = False,
                       calling_function = ''
                       ):
        '''
        Uses the information in channels_spec on what photons to use for correlation
        to extract the needed photons from self.photon_data.
                
        Parameters
        ----------
        channels_spec: 
            Channel configuration specifier for the correlation operation. 
            See description in self.check_channels_spec() for details.
        ext_indices :
            OPTIONAL np.array Externally specified indices of photons in the self.photon_data 
            tttr object to use in correlation. Can be used in slicing 
            data into a series of correlation fucntions, or to 
            randomize stuff (bootstrap). Or the user can exploit it as a handle
            to implement their own custom photon selection logic, obviously.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. If False (default), ones will be 
            imputed (no specific weighting).
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights.
        use_flcs_bg_corr :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from FLCS background correction in photon weights.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the Class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        macro_times_select :
            np.array of macro times of selected photons.
        micro_times_select : 
            np.array of micro times of selected photons.
        weights_select : 
            np.array of weights in channel, which effectily apply the micro time gate.
        indices_select :
            np.array with indices of selected photons

        '''
        
        # Make sure that correlation parameters are complete
        if not self._parameters_set:
            self.update_params()

        # Run check if channels_spec makes sense
        channels_spec_norm = self.check_channels_spec(channels_spec)
        
        # Unpack channel information
        channels = channels_spec_norm[0]
        micro_time_cutoffs = channels_spec_norm[1][0]
        micro_time_gate_indx = channels_spec_norm[1][1]
        
        # Get photon information
        
        # Macro times
        macro_times = self._macro_times.copy()
        
        # Apply burst correction and anomalous-segment excision, if desired
        if use_burst_removal:
            macro_times -= self._macro_times_correction_bursts
            
        if use_mse_filter:
            macro_times -= self._macro_times_correction_mse_filter
            
        # Micro time - always the same, currently no corrections implemented
        # TODO Low priority Add options to correct micro time channel shift and count-rate dependent IRF shift
        micro_times = self._photon_data.micro_times
            
        
        # Get selection
        mask_select = np.ones(macro_times.shape, dtype = np.bool8)

        # Channel-based selection.
        indices_channels = self._photon_data.get_selection_by_channel(channels)
        mask_temp = np.zeros_like(mask_select) # Default zero
        mask_temp[indices_channels] = True # Ones where we keep
        mask_select *= mask_temp # Apply
        
        # Burst removal, if applicable
        if use_burst_removal:
            mask_select *= self._weights_burst_removal
        
        if use_mse_filter:
            mask_select *= self._weights_mse_filter

        # undrifting, if applicable
        if use_drift_correction:
            mask_select *= self._weights_undrift != 0
        
            
        # Externally specified indices, if applicable
        if ext_indices.shape[0] > 0: 
            mask_temp[:] = False # Reset
            mask_temp[ext_indices] = True # Insert new selection
            mask_select *= mask_temp # Apply
            
        # Weights for FLCS or other custom filtering, if applicable
        if use_ext_weights:
            # Are there any photons we can discard based on custom weights?
            mask_select *= self._weights_ext != 0            
            
            
            
        # Micro time gates, if applicable
        if len(micro_time_cutoffs) > 0: 
            
            # Create gates
            if len(micro_time_cutoffs) == 1: 
                # Single cutoff
                gates = np.array([0, micro_time_cutoffs[0], 1]) * self._n_micro_time_bins
                        
            else: 
                # Muliple cutoffs
                gates = [0.]
                [gates.append(gate) for gate in micro_time_cutoffs]
                gates.append(1.)
                gates = np.array(gates) * self._n_micro_time_bins
            
            
            # Find which photons are in gates
            if len(micro_time_gate_indx) == 1: 
                
                # Use only one gate: Simply find photons that are inside the gate
                selection_micro_time = np.logical_and(micro_times >= gates[micro_time_gate_indx[0]],
                                                      micro_times <= gates[micro_time_gate_indx[0] + 1])
                
            else: # Use multiple gates
            
                # Start by creating an all-false array
                selection_micro_time = np.zeros((self._n_total_photons), dtype = np.bool8)
                
                # Fill up with photons that fall into each gate
                for gate_index in micro_time_gate_indx:
                    selection_micro_time = np.logical_or(selection_micro_time, np.logical_and(micro_times >= gates[gate_index],
                                                                                              micro_times <= gates[gate_index + 1]))
            
            # Apply micro time gate
            mask_temp[:] = False # Reset
            mask_temp[selection_micro_time] = True # Insert new selection
            mask_select *= mask_temp # Apply
            
        # Now that we have looked at all selection criteria, we apply selection to photon information
        indices_select = np.nonzero(mask_select)[0]
        macro_times_select = macro_times[indices_select]
        micro_times_select = micro_times[indices_select]
        
        if use_ext_weights:
            weights_select = self._weights_ext[indices_select]
            
        else:
            weights_select = np.ones_like(macro_times_select, dtype = float)
        
        if use_drift_correction:
            weights_select *= self._weights_undrift[indices_select]
        
        if use_flcs_bg_corr:
            weights_select *= self._weights_flcs_bg_corr[indices_select]
            
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''select_photons: Get photons for processing and applying corrections.''',
                                  log_message = f'''Parameters used:
                                    channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm}
                                    ext_indices specified: {ext_indices.shape[0] > 0}
                                    use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                                    use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
                                    use_flcs_bg_corr (use of FLCS-based background subtraction): {use_flcs_bg_corr}
                                    use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                                    use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                                    
                                    overall number of photons selected: {indices_select.shape[0]}
                                    ''',
                                    calling_function = calling_function)

        return macro_times_select, micro_times_select, weights_select, indices_select


    def afterpulse_correlation(self, 
                                lag_times, 
                                acr, 
                                channels_spec,
                                suppress_logging = False,
                                calling_function = ''):
        '''
        Predict the afterpulsing pattern for a given detector 
        channel (for subtraction). This considers afterpulsing 
        subtraction according to the number of discarded 
        pseudo-photons (afterpulses) assuming them to have an 
        equal distribution in microtime data, which is typically reasonable.
        
        Regarding PIE gates, the "second photon" quasi-channel is normally the 
        relevant one for afterpulsing, in case one correlates distinct 
        time gates from the same channel
        
        IMPUTS:
            lag_times:
                1D np.array of correlation lag times for which to predict AP in ns
            acr:
                Average count rate for relevant channel in 1/ns
            channels_spec: 
                Channel configuration specifier for the correlation operation. 
                See description in self.check_channels_spec() for details.
            suppress_logging :
                OPTIONAL bool. If the function called with suppress_logging == True, the call will 
                not be registered in the log file even if the class instance has been 
                set to create a log file. The default is False.
            calling_function : 
                string, optional This is a handle specifically meant for logging 
                which function had called this function. To make the code stack 
                more understandable in the log.

        OUTPUTS:
            G_afterpulse:
                np.array of same size as lag_times. Contains prediction for normalized 
                afterpulsing correlation.
            
        '''
        
        # Make sure that correlation parameters are complete
        if not self._parameters_set:
            self.update_params()

        # Unpack required channels information
        channels_spec_norm = self.check_channels_spec(channels_spec)
        
        # Nested tuple structure
        channel = channels_spec_norm[0]
        micro_time_cutoffs = channels_spec_norm[1][0]
        micro_time_gate_indx = channels_spec_norm[1][1]
        
        if len(channel) > 1:
            raise ValueError('channels_spec for afterpulse_correlation() must contain only one channel. Calibration-based afterpulsing subtraction is currently only supported for single channel autocorrelation, not for sum channels.')
        
        # Unpack relevant detector afterpulsing characteristics
        detector_params = self._afterpulsing_params[channel, :]
        ap_amp_1 = detector_params[0, 0]
        ap_tau_1 = detector_params[0, 1]
        ap_amp_2 = detector_params[0, 2]
        ap_tau_2 = detector_params[0, 3]
        afterpulse_p = self._afterpulsing_p[channel]

        # Get scaling factor for used pseudo-photons discarded by micro time gating
        if len(micro_time_cutoffs) == 0:
            # Empty tuple: Use all
            micro_time_used_fraction = 1.
        
        else: # There is some gating used
            if len(micro_time_cutoffs) == 1: 
                # Single cutoff
                gates = np.array([0, micro_time_cutoffs[0], 1])
                        
            else: 
                # Muliple cutoffs
                gates = np.array(0)
                
                for gate in micro_time_cutoffs:
                    gates.append(gate)
                    
                gates.append(1)
            
            micro_time_used_fraction = 0.
            if type(micro_time_gate_indx) == int: 
                micro_time_used_fraction += gates[micro_time_gate_indx+1] - gates[micro_time_gate_indx]
                
            else:
                for gate_index in micro_time_gate_indx:
                    micro_time_used_fraction += gates[gate_index+1] - gates[gate_index]
                    
        # Actual afterpulsing model: The subtraction uses a biexponential model
        G_afterpulse = ((ap_amp_1 * np.exp(-lag_times/ap_tau_1) + ap_amp_2 * np.exp(-lag_times/ap_tau_2)) # Afterpulsing correlation numerator
                        / (1 + afterpulse_p) / acr # Afterpulsing correlation denominator
                        * micro_time_used_fraction) # Correction for micro time gates discarding part of the AP pseudo-photons
        
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''afterpulse_correlation: Created afterpulsing prediction using biexponential model''',
                                  log_message = f'''Expression:
                                      (A1 * exp(-t/tau1) + A2 * exp(-t/tau2)) * micro_time_gate_correction / (1 + (A1 * tau1 + A2 * tau2)) / count_rate
                                      Parameters used:
                                      channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm}
                                      A1 [GHz]: {ap_amp_1}
                                      tau1 [ns]: {ap_tau_1}
                                      A2 [GHz]: {ap_amp_2}
                                      tau2 [ns]: {ap_tau_2}
                                      count rate [GHz]: {acr}
                                      amplitude correction factor for afterpulses discarded by micro time gating: {micro_time_used_fraction}
                                      ''',
                                      calling_function = calling_function)

        return G_afterpulse


    def correlation_apply_filters(self, 
                                channels_spec_1, 
                                channels_spec_2,
                                ext_indices = np.array([]),
                                tau_min = None, 
                                tau_max = None,
                                use_ext_weights = False, 
                                use_drift_correction = False, 
                                use_flcs_bg_corr = False,
                                use_burst_removal = False,
                                use_mse_filter = False,
                                suppress_logging = False,
                                calling_function = ''):
        
        ''' 
        Perform a correlation function calculation with possibility to apply all
        filters built into this class.
        
        INPUTS:
            channels_spec_1, channels_spec_2: 
                Two channel configuration specifiers for the correlation operation. 
                See description in self.check_channels_spec() for details.
            ext_indices :
                OPTIONAL np.array Externally specified indices of photons in the self.photon_data 
                tttr object to use in correlation. Can be used in slicing 
                data into a series of correlation fucntions, or to 
                randomize stuff (bootstrap), but of course this is also a handle 
                for the user to do their own custom photon selection on top of 
                built-in filters.
            tau_min, tau_max:
                OPTIONAL specifier for minimum and maximum lag time to correlate. 
                float in ns. If empty or not specified at all, will be replaced by global defaults.
            use_ext_weights :
                OPTIONAL bool. Whether to use the external weights stored in 
                self._weights_ext containing photon weights from FLCS 
                filter etc. If False (default), ones will be 
                imputed (no specific weighting).
            use_drift_correction :
                OPTIONAL bool with default False. Whether to consider photon weights 
                from undrifting in photon weights.
            use_flcs_bg_corr :
                OPTIONAL bool with default False. Whether to consider photon weights 
                from FLCS background correction in photon weights.
            use_burst_removal :
                OPTIONAL bool. Specifies whether or not to use the attributes 
                self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
                photons labelled as burst photons
            use_mse_filter :
                OPTIONAL bool. Specifies whether or not to use the attributes 
                self._weights_anomalous_segments and self._macro_times_correction_mse_filter
                to mask out photons labelled as being in an anomalous time segment.
            suppress_logging :
                OPTIONAL bool. If the function called with suppress_logging == True, the call will 
                not be registered in the log file even if the class instance has been 
                set to create a log file. The default is False.
            calling_function : 
                string, optional This is a handle specifically meant for logging 
                which function had called this function. To make the code stack 
                more understandable in the log.
                
        OUTPUTS:
            lag_times: 
                correlation lag times (x axis) in ns
            cc: 
                Correlation function (with +1 offset)
            acr1, acr2:
                Average count rates in the correlation channels, 
                considering micro time gates. In inverse ns.

        '''
        
        # Make sure that correlation parameters are complete
        if not self._parameters_set:
            self.update_params()
                
        # Run input check for limits of correlation times to use.
        tau_min, tau_max = self.check_tau_min_max(tau_min, tau_max)

        # Unpack required channels information
        channels_spec_norm_ch1 = self.check_channels_spec(channels_spec_1)
        channels_spec_norm_ch2 = self.check_channels_spec(channels_spec_2)
        
        # Get photons
        macro_times_ch1, micro_times_ch1, weights_ch1, _ = self.select_photons(channels_spec_norm_ch1, 
                                                                                use_ext_weights = use_ext_weights,
                                                                                ext_indices = ext_indices,
                                                                                use_drift_correction = use_drift_correction,
                                                                                use_flcs_bg_corr = use_flcs_bg_corr,
                                                                                use_burst_removal = use_burst_removal,
                                                                                use_mse_filter = use_mse_filter,
                                                                                suppress_logging = suppress_logging,
                                                                                calling_function = 'correlation_apply_filters')
        
        macro_times_ch2, micro_times_ch2, weights_ch2, _ = self.select_photons(channels_spec_norm_ch2, 
                                                                                use_ext_weights = use_ext_weights,
                                                                                ext_indices = ext_indices,
                                                                                use_drift_correction = use_drift_correction,
                                                                                use_flcs_bg_corr = use_flcs_bg_corr,
                                                                                use_burst_removal = use_burst_removal,
                                                                                use_mse_filter = use_mse_filter,
                                                                                suppress_logging = suppress_logging,
                                                                                calling_function = 'correlation_apply_filters')
        
        # Remove possible offset from actual measurement start
        window_start = np.min([np.min(macro_times_ch1), np.min(macro_times_ch2)])
        macro_times_ch1 -= window_start
        macro_times_ch2 -= window_start
                
        # Average count rates with micro time gates applied, in inverse ns
        acr_1 = np.sum(weights_ch1) / np.max(macro_times_ch1) / self._macro_time_resolution
        acr_2 = np.sum(weights_ch2) / np.max(macro_times_ch2) / self._macro_time_resolution

        # Perform correlation         
        corr = tttrlib.Correlator()
        corr.n_bins = self._sampling
        corr.n_casc = int(self._n_casc) 
        corr.method = self._correlation_method
        corr.set_events(macro_times_ch1, 
                        weights_ch1, 
                        macro_times_ch2, 
                        weights_ch2)
        
        if self._micro_time_corr:
            corr.make_fine = True
            corr.set_microtimes(micro_times_ch1,
                                micro_times_ch2, 
                                self._n_micro_time_bins)
            
        cc = corr.get_corr_normalized()
        lag_times = corr.get_x_axis_normalized() * self._macro_time_resolution

        # Further treatment of cc
        # Start by unpacking channel_config further for the following logic
        channel_1 = channels_spec_norm_ch1[0]
        micro_time_gates_1 = channels_spec_norm_ch1[1]
        channel_2 = channels_spec_norm_ch2[0]
        micro_time_gates_2 = channels_spec_norm_ch2[1]
        
        if self._cross_corr_symm and channel_1 != channel_2: 
            # Two-channel cross-correlation with symmetric time: Average forward 
            # and reverse cc
            corr_rev = tttrlib.Correlator()
            corr_rev.n_bins = self._sampling
            corr_rev.n_casc = int(self._n_casc)
            corr_rev.method = self._correlation_method
            corr_rev.set_events(macro_times_ch2, 
                            weights_ch2, 
                            macro_times_ch1, 
                            weights_ch1)
            
            if self._micro_time_corr:
                corr_rev.make_fine = True
                corr_rev.set_microtimes(micro_times_ch2,
                                        micro_times_ch1, 
                                        self._n_micro_time_bins)  
                
            cc_rev = corr_rev.get_corr_normalized()
            # Average forward and backward cc, and subtract +1 offset
            cc_processed = (cc + cc_rev) / 2 
            cc_processed -= 1
            comment_string = 'Calculated as two-channel cross-correlation function with assumption of time symmetry.' # for log

            
        elif self._cross_corr_symm and channel_1 == channel_2 and micro_time_gates_1 != micro_time_gates_2: 
            # Time-symmetric cross-correlation of distinct micro
            # time gates within the same channel:
            # Subtract afterpulsing, average forward and backward
            # cc, and subtract +1 offset (AP subtraction is dummy
            # 0 subtraction if subtract_afterpulsing == False)
            corr_rev = tttrlib.Correlator()
            corr_rev.n_bins = self._sampling
            corr_rev.n_casc = int(self._n_casc)
            corr_rev.method = self._correlation_method
            corr_rev.set_events(macro_times_ch2, 
                            weights_ch2, 
                            macro_times_ch1, 
                            weights_ch1)
            
            if self._micro_time_corr:
                corr_rev.make_fine = True
                corr_rev.set_microtimes(micro_times_ch2,
                                        micro_times_ch1, 
                                        self._n_micro_time_bins)    
                
            cc_rev = corr_rev.get_corr_normalized()
            
            if self._subtract_afterpulsing and not use_flcs_bg_corr:
                cc -= self.afterpulse_correlation(lag_times, 
                                                  acr_2, 
                                                  channels_spec_norm_ch2, 
                                                  suppress_logging = suppress_logging,
                                                  calling_function = 'correlation_apply_filters (forward call)')
                
                cc_rev -= self.afterpulse_correlation(lag_times, 
                                                      acr_1, 
                                                      channels_spec_norm_ch1, 
                                                      suppress_logging = suppress_logging,
                                                      calling_function = 'correlation_apply_filters (backward call)')
                
            cc_processed = (cc + cc_rev) / 2 -1
            comment_string = 'Calculated as cross-correlation function betwen different micro time gates within the same channel, with assumption of time symmetry.' # for log

        elif not self._cross_corr_symm and channel_1 == channel_2 and micro_time_gates_1 != micro_time_gates_2: 
            # Cross-correlation of distinct micro
            # time gates within the same channel, without time symmetry:
            # Subtract afterpulsing and subtract +1 offset (AP subtraction is dummy
            # 0 subtraction if subtract_afterpulsing == False)
            
            if self._subtract_afterpulsing and not use_flcs_bg_corr:
                cc_processed = (cc - self.afterpulse_correlation(lag_times, 
                                                                 acr_2, 
                                                                 channels_spec_norm_ch2, 
                                                                 suppress_logging = suppress_logging,
                                                                 calling_function = 'correlation_apply_filters')) - 1 
                
            else:
                cc_processed = cc - 1
                
            comment_string = 'Calculated as cross-correlation function betwen different micro time gates within the same channel, without assumption of time symmetry.' # for log

        elif channel_1 == channel_2 and micro_time_gates_1 == micro_time_gates_2: 
            # Autocorrelation with equal microtime gates 
            # (possibly none): Subtract afterpulsing (dummy 
            # subtraction if subtract_afterpulsing == False). 
            if self._subtract_afterpulsing:
                cc_processed = (cc - self.afterpulse_correlation(lag_times, 
                                                                 acr_2, 
                                                                 channels_spec_norm_ch2, 
                                                                 suppress_logging = suppress_logging,
                                                                 calling_function = 'correlation_apply_filters')) - 1 
                
            else:
                cc_processed = cc - 1
                
            comment_string = 'Calculated as auto-correlation function.' # for log

        else: 
            # Two-channel cross-correlation, but no time symmetry assumption: 
            # Nothing specific to do, just subtract +1 offset
            cc_processed = cc - 1
            
            comment_string = 'Calculated as two-channel cross-correlation function without assumption of time symmetry.' # For log
            
        # Crop to queried lag time range, and return
        keep = np.logical_and(lag_times >= tau_min, lag_times <= tau_max)  
        lag_times = lag_times[keep]
        cc_processed = cc_processed[keep]
        
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''correlation_apply_filters: Calculate correlation function applying various filters''',
                                  log_message = f'''Parameters used:
                                    channels_spec_1, channels_spec_2 ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm_ch1}, {channels_spec_norm_ch2}
                                    ext_indices specified: {ext_indices.shape[0] > 0}
                                    tau_min, tau_max (lower and upper bounds of correlation time) [ns]: {tau_min}, {tau_max}
                                    -> Actual shortest and longest lag times calculated [ns]: {lag_times[0]}, {lag_times[-1]}.
                                    use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                                    use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
                                    use_flcs_bg_corr (use of FLCS-based background subtraction): {use_flcs_bg_corr}
                                    use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                                    use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                                    
                                    Average count rates (considering corrections and weights) [GHz]:
                                    First correlation channel: {acr_1}
                                    Second correlation channel: {acr_2}
                                    
                                    ''' + comment_string + '''
                                    ''',
                                    calling_function = calling_function)
                                      
        return lag_times, cc_processed, acr_1 ,acr_2



    def get_segment_ccs(self,
                        channels_spec_1,
                        channels_spec_2,
                        minimum_window_length,
                        tau_min = None,
                        tau_max = None,
                        use_ext_weights = False,
                        use_drift_correction = False,
                        use_flcs_bg_corr = False,
                        use_burst_removal = False,
                        use_mse_filter = False,
                        suppress_logging = False,
                        calling_function = ''
                        ):
        '''
        Calculate (corrected) correlation functions for equal-sized time segments
        of the measurement, e.g. for analyzing trends over time, for getting 
        statistics about the spread of parameters, for estimating uncertainty, ...

        Parameters
        ----------
        channels_spec_1, channels_spec_2: 
            Two channel configuration specifiers for the correlation operation. 
            See description in self.check_channels_spec() for details.
        minimum_window_length:
            OPTIONAL float > tau_max. Specifies minimum width for the windows 
            in which to separate the trace. In seconds.
        tau_min, tau_max:
            OPTIONAL specifier for minimum and maximum lag time to correlate. 
            float in ns. If empty or not specified at all, will be replaced by global defaults.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. OPTIONAL. If False (default), ones will be 
            imputed (no specific weighting).
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights.
        use_flcs_bg_corr :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from FLCS background correction in photon weights.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.


        Returns
        -------
        lag_times: np.array 
            correlation lag times (x axis) in ns
        segment_ccs: np.array (2D)
            Correlation functions of all time segments. Axis 0 is of same length 
            as lag_times, axis 1 of length equal number of segments into which 
            time axis was segmented. It can happen that the correlation fails 
            for some segments (rare, reason unclear to me). Those correlation 
            functions are simply returned as columns of zeros.
        usable_segments : list of int
            indices in axis 1 of segment_ccs denoting in which cases the 
            correlation function calculation was successful.
        start_stop : np.array (2D)
            photon indices that correspond to start and stop of each segment. 
            Axis 0 is the iteration over segments, in axis 1, element 0 is 
            start index, element 1 is stop index. Returned for cases where more 
            detailled inspection of the correlation function in relation to 
            the position or length of its time trace segment is needed.

        '''
        
        # Make sure that correlation parameters are complete
        if not self._parameters_set:
            self.update_params()

        # Input check
        tau_min, tau_max = self.check_tau_min_max(tau_min, tau_max)
        channel_1, micro_time_gates_1 = self.check_channels_spec(channels_spec_1)
        channel_2, micro_time_gates_2 = self.check_channels_spec(channels_spec_2)
        
        try:
            minimum_window_length *= 1E9 # to ns
            if not(type(minimum_window_length) in [float, np.float32, np.float64] and minimum_window_length > tau_max):
                raise ValueError('Dummy message, this will never be seen')
        except:
            raise ValueError('minimum_window_length must be float in ns > tau_max in ns.')
        
        # Get time windows of segments, considering possible shortening of acquisition time through corrections
        macro_times = self._macro_times.copy()
        
        if use_burst_removal:
            macro_times -= self._macro_times_correction_bursts
            
        if use_mse_filter:
            macro_times -= self._macro_times_correction_mse_filter
            
        time_windows = tttrlib.ranges_by_time_window(macro_times,
                                                     minimum_window_length = minimum_window_length / 1E6,
                                                     macro_time_calibration = self._macro_time_resolution / 1E6
                                                     ) # No clue why the conversion to ms time scale is needed, but somehow time_windows is returned empty without the conversion...
        
        # reshape the time windows array (interleaves -> start stop)
        n_time_windows = len(time_windows)//2
        start_stop = time_windows.reshape((n_time_windows, 2))
        
        # Initialize some empty arrays
        segment_ccs = np.array([])
        usable_segments = []
        segment_index = 0

        # Loop over segments to get segment ccs
        for start, stop in start_stop:
            try:
                # Slice photons that belong to unused channels
                keep_indices = np.arange(start, stop, dtype=np.int64)
                
                # Get segment_cc and add to array 
                lag_times, segment_cc, _, _ = self.correlation_apply_filters((channel_1, micro_time_gates_1),
                                                                             (channel_2, micro_time_gates_2),
                                                                             use_ext_weights = use_ext_weights, 
                                                                             ext_indices = keep_indices,
                                                                             tau_min = tau_min, 
                                                                             tau_max = tau_max,
                                                                             use_drift_correction = use_drift_correction,
                                                                             use_flcs_bg_corr = use_flcs_bg_corr,
                                                                             use_burst_removal = use_burst_removal,
                                                                             use_mse_filter = use_mse_filter,
                                                                             suppress_logging = (not segment_index == 0) or suppress_logging, # if at all, log only one curve to reduce redundancy
                                                                             calling_function = 'get_segment_ccs')
                
                if segment_ccs.shape[0] == 0: # We arrive here when we for the first time successfully calculated a CC, which sometimes fails
                    # We do this here because only now do we know the number of lag_times,
                    # and we do not deterministically do it in iteration 0 as
                    # it could be that iteration 0 fails. Instead, we do it whenever
                    # we for the first time get a segment that does not fail.
                    segment_ccs = np.zeros((segment_cc.shape[0], n_time_windows), dtype = float)
                    
                segment_ccs[:,segment_index] = segment_cc
                usable_segments.append(segment_index)

            except:
                # Something went wrong with this segment, for whatever reason. 
                # Nothing to do about that, other than discreetly flagging 
                # it as a segment where something's not right and that should 
                # therefore be discarded by NOT labelling this as a usable 
                # segment to be used in the following calculation steps...
                pass
            
            finally:
                # In either case, iterate to next segment
                segment_index += 1
        
        # Convert to array now
        usable_segments = np.array(usable_segments)
        
        if self._write_results:
            # Write results to csv and png output
            out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
            self._out_file_counter += 1
            
            # Putting together name requires some logic
            if channel_1 == channel_2 and micro_time_gates_1 == micro_time_gates_2:
                # "Proper" autocorrelation, with the same micro time configuration
                 out_path_full += '_segment_ACFs_ch' + ''.join([str(element) for element in channel_1])
                
            elif channel_1 != channel_2 and self._cross_corr_symm:
                # Cross-correlation assuming time symmetry
                 out_path_full += '_segment_CCFs_symm_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
                
            elif channel_1 == channel_2 and micro_time_gates_1 != micro_time_gates_2 and not self._cross_corr_symm:
                # Cross-correlation of distinct micro time bins within the same channel
                 out_path_full += '_segment_Microt_CCFs_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
                
            elif channel_1 == channel_2 and micro_time_gates_1 != micro_time_gates_2 and self._cross_corr_symm: 
                # Cross-correlation of distinct micro time bins within the same channel, assuming time symmetry
                 out_path_full += '_segment_Microt_CCFs_symm_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
                
            else:
                # Cross-correlation not assuming time symmetry
                 out_path_full += '_segment_CCFs_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
            
            # Update name according to applied corrections
            out_path_full += ('_br' if use_burst_removal else '') + \
                             ('_dt' if use_drift_correction else '') + \
                             ('_ar' if use_mse_filter else '') + \
                             ('_w' if use_ext_weights else '') + \
                             ('_bg' if use_flcs_bg_corr else '')
            
            # Plot and save figure
            fig, ax = plt.subplots(nrows=1, ncols=1)
            
            # Cycle through colors
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = cycle(prop_cycle.by_key()['color'])

            for segment_index in usable_segments:
                iter_color = next(colors)
                ax.semilogx(lag_times*1E-9, 
                            segment_ccs[:,segment_index],
                            marker = '', 
                            linestyle = '-', 
                            alpha = 0.7,
                            color = iter_color)
                
            ax.set_title('Segment correlation functions') 
            fig.supxlabel('Correlation time [s]')
            fig.supylabel('G(\u03C4)')
            ax.set_xlim(lag_times.min() * 1E-9, lag_times.max() * 1E-9)
            plot_y_min_max = (np.percentile(segment_ccs[:,usable_segments], 3), np.percentile(segment_ccs[:,usable_segments], 97))
            ax.set_ylim(plot_y_min_max[0] / 1.2 if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
                        plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)

            plt.savefig(out_path_full + '.png', dpi=300)
            plt.close()

            # Create and save spreadsheet
            out_table = pd.DataFrame(data ={'Lagtime[s]':lag_times*1E-9})
            
            for segment_index in range(segment_ccs.shape[1]):
                out_table['CF_segment' + str(segment_index)] = segment_ccs[:, segment_index]
                
            out_table.to_csv(out_path_full + '.csv', 
                             index = False, 
                             header = self._include_header)

        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''get_segment_ccs: Calculate correlation functions with filters for evenly spaces time segments of the measurement.''',
                                  log_message = f'''Parameters used:
                                    channels_spec_1, channels_spec_2 ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {(channel_1, micro_time_gates_1)}, {(channel_2, micro_time_gates_2)}
                                    minimum_window_length (shortest allowed window length) [s]: {minimum_window_length*1E-9}
                                    tau_min, tau_max (lower and upper bounds of correlation time) [ns]: {tau_min}, {tau_max}
                                    use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                                    use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
                                    use_flcs_bg_corr (use of FLCS-based background subtraction): {use_flcs_bg_corr}
                                    use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                                    use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                                    
                                    Split time trace of length {np.round(self._acquisition_time*1E-9, 1)} s into {start_stop.shape[0]} segments, of which {usable_segments.shape[0]} were usable.
                                    ''' + (f'''Wrote results to {out_path_full}.csv/png''' if self._write_results else ''),
                                    calling_function = calling_function)         
        # Done
        return lag_times, segment_ccs, usable_segments, start_stop



    def get_Wohland_SD(self, 
                       channels_spec_1,
                       channels_spec_2,
                       minimum_window_length = [],
                       tau_max = None,
                       tau_min = None,
                       use_ext_weights = False, 
                       use_drift_correction = False, 
                       use_flcs_bg_corr = False,
                       use_burst_removal = False,
                       use_mse_filter = False,
                       suppress_logging = False,
                       calling_function = ''
                       ):
        '''
        Wohland method of uncertainly standard deviation calculation for a correlation function.
        Splits the photon trace into a number of equal-length segments, correlates 
        those, and calculates uncertainty for the full-length curve from the 
        standard deviation between segments.
        
        Based on Wohland, Vogel, Rigler Biophysical J 2001, DOI: 10.1016/S0006-3495(01)76264-9 (method 4 proposed in that paper)
        
        Note that ext_indices which you can use for many other methods is 
        currently incompatible with this function. Use weights_external 
        attribute to work around that, if needed.

        Parameters
        ----------
        channels_spec_1, channels_spec_2: 
            Two channel configuration specifiers for the correlation operation. 
            See description in self.check_channels_spec() for details.
        minimum_window_length:
            OPTIONAL float > tau_max. Specifies minimum width for the windows 
            in which to separate the trace. In seconds.
        tau_min, tau_max:
            OPTIONAL specifier for minimum and maximum lag time to correlate. 
            float in ns. If empty or not specified at all, will be replaced by global defaults.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. OPTIONAL. If False (default), ones will be 
            imputed (no specific weighting).
        use_flcs_bg_corr :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from FLCS background correction in photon weights.
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        sd_cc : 
            np.array with uncertainty SD for each correlation bin.

        '''
        # Make sure that correlation parameters are complete
        if not self._parameters_set:
            self.update_params()

        # Run input check for limits of correlation times to use.
        tau_min, tau_max = self.check_tau_min_max(tau_min, tau_max)
        channels_spec_norm_ch1 = self.check_channels_spec(channels_spec_1)
        channels_spec_norm_ch2 = self.check_channels_spec(channels_spec_2)
        
        # Get acquistion time from macro_time information, considering possible shortening of acquisition time through corrections
        effective_acquisition_time = self._macro_times[-1]
        
        if use_burst_removal:
            effective_acquisition_time -= self._macro_times_correction_bursts[-1]
            
        if use_mse_filter:
            effective_acquisition_time -= self._macro_times_correction_mse_filter[-1]

        effective_acquisition_time *= self._macro_time_resolution # to ns
        
        if type(minimum_window_length) in [list, np.ndarray] and len(minimum_window_length) == 0:
            # Auto-choose
            minimum_window_length = np.max([effective_acquisition_time/10., 5.*tau_max]).astype(np.float64) *1E-9 # to seconds for consistency

        elif type(minimum_window_length) in [float, np.float32, np.float64] and minimum_window_length*1E9 > tau_max:
            minimum_window_length = np.float64(minimum_window_length)
            
        else:
            raise ValueError('minimum_window_length must be float or empty list. minimum_window_length must be > tau_max (keep in mind that tau_max in this function is handled as being given in nanoseconds, minimum_window_length as being given in seconds!')
            
        log_comment = f'''Parameters used:
            channels_spec_1, channels_spec_2 ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm_ch1}, {channels_spec_norm_ch2}
            minimum_window_length (shortest allowed window length) [s]: {minimum_window_length}
            tau_min, tau_max (lower and upper bounds of correlation time) [ns]: {tau_min}, {tau_max}
            use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
            use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
            use_flcs_bg_corr (use of FLCS-based background subtraction): {use_flcs_bg_corr}
            use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
            use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
            ''' # This function can end up with three different log outputs, but this is the part they all share.
                    
        # At least 5 windows would be nice for Wohland method SD calculation. 
        # If that is not fulfilled, we use bootstrap as a backup strategy, passing on the exact same settings.
        if effective_acquisition_time * 1E-9 / minimum_window_length < 5: 
            
            # Write to log, if desired
            if self._write_log and not suppress_logging:
                self.write_to_logfile(log_header = '''get_Wohland_SD: Calculate correlation functions uncertainty using Wohland method.''',
                                      log_message = log_comment + '''
                                          
                                        Calculation aborted and instead switched to bootstrap method uncertainty calculation due to too-short acquisition time relative to specified minimum window length.
                                        ''')
                                          
            # Proceed to bootstrapping
            sd_cc = self.get_bootstrap_SD(channels_spec_norm_ch1,
                                          channels_spec_norm_ch2, 
                                          n_bootstrap_reps=10,
                                          use_ext_weights = use_ext_weights, 
                                          tau_min = tau_min,
                                          tau_max = tau_max,
                                          use_drift_correction = use_drift_correction,
                                          use_flcs_bg_corr = use_flcs_bg_corr,
                                          use_burst_removal = use_burst_removal,
                                          use_mse_filter = use_mse_filter,
                                          suppress_logging = suppress_logging,
                                          calling_function = 'get_Wohland_SD')

        else:
            # We get enough windows.

            _, segment_ccs, usable_segments, _ = self.get_segment_ccs(channels_spec_norm_ch1,
                                                                      channels_spec_norm_ch2,
                                                                      minimum_window_length,
                                                                      tau_min = tau_min,
                                                                      tau_max = tau_max,
                                                                      use_ext_weights = use_ext_weights,
                                                                      use_drift_correction = use_drift_correction,
                                                                      use_flcs_bg_corr = use_flcs_bg_corr,
                                                                      use_burst_removal = use_burst_removal,
                                                                      use_mse_filter = use_mse_filter,
                                                                      suppress_logging = suppress_logging,
                                                                      calling_function = 'get_Wohland_SD')
            
            if segment_ccs.shape[1] > 1:
                # At least two segments actually worked, get uncertainty from those
            
                var_segment_ccs = np.var(segment_ccs[:,usable_segments], axis = 1)
                
                # As we are dealing with segments, we need to scale the variance 
                # to the full acquisition time.
                # We do that under the assumption that the uncertainty (standard error)
                # decreases with the square root of the acquisition time.
                n_time_windows = segment_ccs.shape[1]                
                sd_cc = np.sqrt(var_segment_ccs / (n_time_windows - 1))

                # Write to log, if desired
                if self._write_log and not suppress_logging:
                    self.write_to_logfile(log_header = '''get_Wohland_SD: Calculate correlation functions uncertainty using Wohland method.''',
                                      log_message = log_comment + f'''
                                          
                                            Used {usable_segments.shape[0]} segments for uncertainty estimation.
                                            ''',
                                            calling_function = calling_function)

            else: 
                # Complete failure of Wohland method for whatever reason: 
                # Try bootstrap instead with the exact same settings.
                
                # Write to log, if desired
                if self._write_log and not suppress_logging:
                    self.write_to_logfile(log_header = '''get_Wohland_SD: Calculate correlation functions uncertainty using Wohland method.''',
                                      log_message = log_comment + '''
                                          
                                        Wohland method uncertainty estimation failed as not enough segments yielded usable correlation functions. Switching to bootstrap method instead.
                                        ''',
                                        calling_function = calling_function)
                                              
                # Proceed to bootstrapping
                sd_cc = self.get_bootstrap_SD(channels_spec_1,
                                              channels_spec_2, 
                                              n_bootstrap_reps=10,
                                              use_ext_weights = use_ext_weights, 
                                              tau_min = tau_min,
                                              tau_max = tau_max,
                                              use_drift_correction = use_drift_correction,
                                              use_flcs_bg_corr = use_flcs_bg_corr,
                                              use_burst_removal = use_burst_removal,
                                              use_mse_filter = use_mse_filter,
                                              suppress_logging = suppress_logging,
                                              calling_function = 'get_Wohland_SD')

        return sd_cc


    def get_bootstrap_SD(self,
                            channels_spec_1,
                            channels_spec_2, 
                            n_bootstrap_reps = 10,
                            tau_min = None,
                            tau_max = None,
                            use_ext_weights = False, 
                            use_drift_correction = False,
                            use_flcs_bg_corr = False,
                            use_burst_removal = False,
                            use_mse_filter = False,
                            suppress_logging = False,
                            calling_function = ''):
        
        '''
        Bootstrap-resampling-based uncertainly standard deviation calculation for 
        a correlation function. Normally, bootstrapping works best for unsorted 
        data, and is not really a method for ordered time series data. It does 
        work surprisingly well for FCS though if you bootstrap which photons 
        to include in the correlation function, without changing anything about 
        their order.
        
        Bootstrapping yields slightly different uncertainty estimates than the 
        Wohland method. In essence, bootstrapping only captures the uncertaintly 
        from photon shot noise, not that from insufficient particle statistics. 
        The advantages of bootstrapping are that, different from the Wohland method, 
        it works with arbitrarily short measurements, and, different from other 
        methods, it is free from any model assumptions about the underlying
        physical processes.
        
        Therefore, it is included here mostly for use as a backup method in 
        case the data is too short for a meaningful Wohland method uncertainty 
        calculation.
        
        Note that ext_indices which you can use for many other methods is 
        currently incompatible with this function. Use weights_external 
        attribute to work around that, if needed.
        
        Parameters
        ----------
        channels_spec_1, channels_spec_2: 
            Two channel configuration specifiers for the correlation operation. 
            See description in self.check_channels_spec() for details.
        n_bootstrap_reps:
            OPTIOANL int > 1. How many bootstrap replicates to perform? Defaults to 10.
        tau_min, tau_max:
            OPTIONAL specifier for minimum and maximum lag time to correlate. 
            float in ns. If empty or not specified at all, will be replaced by global defaults.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. If False (default), ones will be 
            imputed (no specific weighting). 
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights.
        use_flcs_bg_corr :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from FLCS background correction in photon weights.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        sd_cc : 
            np.array with uncertainty SD for each correlation bin.

        '''        
        # Make sure that correlation parameters are complete
        if not self._parameters_set:
            self.update_params()
        
        # Input check
        if not (type(n_bootstrap_reps) in [int, np.uint8, np.uint16, np.uint32, np.uint64] and n_bootstrap_reps > 1):
            raise ValueError('n_bootstrap_reps must be int > 1')
            
        tau_min, tau_man = self.check_tau_min_max(tau_min, tau_max)
        channels_spec_norm_ch1 = self.check_channels_spec(channels_spec_1)
        channels_spec_norm_ch2 = self.check_channels_spec(channels_spec_2)

        # Initialize RNG
        rng = np.random.default_rng()
        
        # Initialize results arrays
        sum_bs_cc = np.array([])
        sum_of_squares_bs_cc = np.array([])
        failed_segments = 0
        
        # Loop over segments
        for i_rep in range(n_bootstrap_reps):
            try:
                # Resample photon data
                resample_indxs = rng.choice(self._n_total_photons, size=self._n_total_photons)
                resample_indxs_sort = np.sort(resample_indxs) #For some reason an in-place sort via the np.ndarray.sort() method does not work, so we need an extra step

                # Get rep_cc and add to sum 
                lag_times, rep_cc, _, _ = self.correlation_apply_filters(channels_spec_norm_ch1,
                                                                         channels_spec_norm_ch2,
                                                                         use_ext_weights = use_ext_weights, 
                                                                         ext_indices = resample_indxs_sort,
                                                                         tau_min = tau_min, 
                                                                         tau_max = tau_max,
                                                                         use_drift_correction = use_drift_correction,
                                                                         use_flcs_bg_corr = use_flcs_bg_corr,
                                                                         use_burst_removal = use_burst_removal,
                                                                         use_mse_filter = use_mse_filter,
                                                                         suppress_logging = suppress_logging or sum_bs_cc.shape[0] > 0, # Log only first one, if at all
                                                                         calling_function = 'get_bootstrap_SD')
                
                if sum_bs_cc.shape[0] == 0: # Only initialized, nothing in there yet
                    sum_bs_cc = np.zeros_like(rep_cc, dtype = float)
                    sum_of_squares_bs_cc = np.zeros_like(rep_cc, dtype = float)
                    
                sum_bs_cc += rep_cc 
                sum_of_squares_bs_cc += np.square(rep_cc)
                
            except:
                # Something went wrong with this replicate, for whatever reason. Ignore and only keep track of how many failed.
                failed_segments += 1
                
        # Correct n_time_windows (which will now be used as a normalization factor) for failed_segments
        n_bootstrap_reps_corr = n_bootstrap_reps - failed_segments
        
        if n_bootstrap_reps_corr > 1: # At least two worked - we can use that
            # Get full-length cc's standard deviation from linear sum and
            # sum of squares. Note that some terms may seem unusual due to 
            # the finite sample size rather than ground truth coverage)
            # There's honestly not too much reason for the somewhat complicated 
            # running-sum and moment-based implementation...But it does not 
            # hurt either. this is mostly something historic I copied over 
            # from code where it did matter more than it does here.
            
            # Normalize with number of segments to retrieve first and second moment:
            bs_cc_squared_1st_mom = np.square(sum_bs_cc) / (np.square(n_bootstrap_reps_corr) - n_bootstrap_reps_corr)
            bs_cc_2nd_mom = sum_of_squares_bs_cc / (n_bootstrap_reps_corr - 1)
            
            # Get variance from 1st and 2nd moments:
            var_bs_cc = bs_cc_2nd_mom - bs_cc_squared_1st_mom
            
            # Get SD from variance
            sd_cc = np.sqrt(var_bs_cc)
            
        else: 
            # Complete failure. As bootstrap is already used as fall-back option
            # in case Wohland methods fails, we really can only return a dummy value at this point.
            sd_cc = np.array(1.)
            
            
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''get_bootstrap_SD: Calculate correlation function uncertainty via photon bootstrapping.''',
                                  log_message = f'''Parameters used:
                                    channels_spec_1, channels_spec_2 ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm_ch1}, {channels_spec_norm_ch2}
                                    n_bootstrap_reps (number of bootstrap replicates to perform) {n_bootstrap_reps}
                                    tau_min, tau_max (lower and upper bounds of correlation time) [ns]: {tau_min}, {tau_max}
                                    use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                                    use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
                                    use_flcs_bg_corr (use of FLCS-based background subtraction): {use_flcs_bg_corr}
                                    use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                                    use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                                      
                                    Out of {n_bootstrap_reps} bootstrap replicates, {n_bootstrap_reps_corr} were successful and could be used for uncertainty estimation.
                                    ''',
                                    calling_function = calling_function)

        return sd_cc


    def get_correlation_uncertainty(self,
                                    channels_spec_1,
                                    channels_spec_2, 
                                    default_uncertainty_method = 'Wohland',
                                    minimum_window_length = [],
                                    n_bootstrap_reps = 10,
                                    tau_min = None,
                                    tau_max = None,
                                    use_ext_weights = False, 
                                    use_drift_correction = False,
                                    use_flcs_bg_corr = False,
                                    use_burst_removal = False,
                                    use_mse_filter = False,
                                    suppress_logging = False,
                                    calling_function = ''):
        
        '''
        Wrapper for self.correlation_apply_filters(), self.get_Wohland_SD(),
        and self.get_bootstrap_SD(), including automated writing of output 
        files in .csv format (in the format used in software developed by the 
        lab of Claus AM Seidel), and .png format.

        Parameters
        ----------
        channels_spec_1, channels_spec_2: 
            Two channel configuration specifiers for the correlation operation. 
            See description in self.check_channels_spec() for details.
        default_uncertainty_method : 
            OPTIONAL string with default 'Wohland'. Alternative is 'Bootstrap'.
            If 'Wohland' is chosen, self.get_Wohland_SD() is used as the default
            method of standard deviation calculation, and self.get_bootstrap_SD()
            as the backup method. If 'Bootstrap' is chosen, the software will 
            directly go to self.get_bootstrap_SD().
        minimum_window_length:
            OPTIONAL float > tau_max. Specifies minimum width for the windows 
            in which to separate the trace in self.get_Wohland_SD(). In seconds.
        n_bootstrap_reps:
            OPTIOANL int > 1 with default 10. How many bootstrap replicates to 
            perform in self.get_bootstrap_SD()?
        tau_min, tau_max:
            OPTIONAL specifiers for minimum and maximum lag time to correlate. 
            float in ns. If empty or not specified at all, will be replaced by global defaults.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. If False (default), ones will be 
            imputed (no specific weighting). 
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights.
        use_flcs_bg_corr :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from FLCS background correction in photon weights.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        lag_times : 
            np.array (1D) with correlation lag times (x axis) in ns
        cc: 
            np.array (1D) with correlation function
        sd_cc : 
            np.array (1D) with uncertainty SD for each correlation bin.
        acr1, acr2:
            Average count rates in the correlation channels, considering micro 
            time gates and filters. In inverse ns.
            
        '''
        # Make sure that correlation parameters are complete
        if not self._parameters_set:
            self.update_params()

        # Input check
        channel_1, micro_time_gates_1 = self.check_channels_spec(channels_spec_1)
        channel_2, micro_time_gates_2 = self.check_channels_spec(channels_spec_2)

        if not default_uncertainty_method in ['Wohland', 'Bootstrap', 'bootstrap']:
            raise ValueError('Invalid input for default_uncertainty_method: Must be "Wohland" or "Bootstrap".')
            
        # Get correlation and uncertainty
        lag_times, cc, acr1, acr2 = self.correlation_apply_filters(channels_spec_1, 
                                                                channels_spec_2,
                                                                tau_min = tau_min, 
                                                                tau_max = tau_max,
                                                                use_ext_weights = use_ext_weights, 
                                                                use_drift_correction = use_drift_correction, 
                                                                use_flcs_bg_corr = use_flcs_bg_corr,
                                                                use_burst_removal = use_burst_removal,
                                                                use_mse_filter = use_mse_filter,
                                                                suppress_logging = suppress_logging,
                                                                calling_function = 'get_correlation_uncertainty')
        
        if default_uncertainty_method == 'Wohland':
            sd_cc = self.get_Wohland_SD(channels_spec_1,
                                        channels_spec_2,
                                        minimum_window_length = minimum_window_length,
                                        tau_max = tau_max,
                                        tau_min = tau_min,
                                        use_ext_weights = use_ext_weights, 
                                        use_drift_correction = use_drift_correction, 
                                        use_flcs_bg_corr = use_flcs_bg_corr,
                                        use_burst_removal = use_burst_removal,
                                        use_mse_filter = use_mse_filter,
                                        suppress_logging = suppress_logging,
                                        calling_function = 'get_correlation_uncertainty')
            
        else:
            # implies default_uncertainty_method == 'Bootstrap' (or lowercase)
            sd_cc = self.get_bootstrap_SD(channels_spec_1,
                                        channels_spec_2, 
                                        n_bootstrap_reps = n_bootstrap_reps,
                                        tau_max = tau_max,
                                        tau_min = tau_min,
                                        use_ext_weights = use_ext_weights, 
                                        use_drift_correction = use_drift_correction, 
                                        use_flcs_bg_corr = use_flcs_bg_corr,
                                        use_burst_removal = use_burst_removal,
                                        use_mse_filter = use_mse_filter,
                                        suppress_logging = suppress_logging,
                                        calling_function = 'get_correlation_uncertainty')
            
        if (sd_cc == 1.).all():
            # Got a dummy value out, meaning that we had some dramatic failure
            # in SD calculation. Use an array of ones instead for unweighted 
            # fitting...
            sd_cc = np.ones_like(cc)
        
        
        if self._write_results:
            # Auto-create csv and png exports
            
            out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
            self._out_file_counter += 1
            
            # Putting together name requires some logic
            if channel_1 == channel_2 and micro_time_gates_1 == micro_time_gates_2:
                # "Proper" autocorrelation, with the same micro time configuration
                 out_path_full += '_ACF_ch' + ''.join([str(element) for element in channel_1])
                
            elif channel_1 != channel_2 and self._cross_corr_symm:
                # Cross-correlation assuming time symmetry
                 out_path_full += '_CCF_symm_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
                
            elif channel_1 == channel_2 and micro_time_gates_1 != micro_time_gates_2 and not self._cross_corr_symm:
                # Cross-correlation of distinct micro time bins within the same channel
                 out_path_full += '_Microt_CCF_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
                
            elif channel_1 == channel_2 and micro_time_gates_1 != micro_time_gates_2 and self._cross_corr_symm: 
                # Cross-correlation of distinct micro time bins within the same channel, assuming time symmetry
                 out_path_full += '_Microt_CCF_symm_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
                
            else:
                # Cross-correlation not assuming time symmetry
                 out_path_full += '_CCF_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
    
            # Update name according to applied corrections
            out_path_full += ('_br' if use_burst_removal else '') + \
                             ('_dt' if use_drift_correction else '') + \
                             ('_ar' if use_mse_filter else '') + \
                             ('_w' if use_ext_weights else '') + \
                             ('_bg' if use_flcs_bg_corr else '')

            # Plot and save
            fig, ax = plt.subplots(nrows=1, ncols=1)
            
            ax.semilogx(lag_times * 1E-9, cc, 'dk')
            ax.semilogx(lag_times * 1E-9, cc + sd_cc, '-k', alpha = 0.7)
            ax.semilogx(lag_times * 1E-9, cc - sd_cc, '-k', alpha = 0.7)
            ax.set_title('Correlation function with uncertainty') 
            fig.supxlabel('Correlation time [s]')
            fig.supylabel('G(\u03C4)')
            ax.set_xlim(lag_times.min() * 1E-9, lag_times.max() * 1E-9)
            plot_y_min_max = (np.percentile(cc, 3), np.percentile(cc, 97))
            ax.set_ylim(plot_y_min_max[0] / 1.2 if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
                        plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)

            plt.savefig(out_path_full + '.png', dpi=300)
            plt.close()
            
            # Wrap stuff for file writing, and perform CSV export
            acr_col = np.zeros_like(lag_times)
            acr_col[:3] = np.array([acr1 * 1E9, acr2 * 1E9, self._acquisition_time * 1E-9])
            out_table = pd.DataFrame(data = {'Lagtime[s]':lag_times * 1E-9, # from ns to s
                                             'Correlation': cc,
                                             'ACR[Hz]': acr_col,
                                             'Uncertainty_SD': sd_cc})
            
            out_table.to_csv(out_path_full + '.csv', 
                             index = False, 
                             header = self._include_header)

        # Write to log, if desired
        if self._write_log and not suppress_logging:
            
            log_message = f'''Parameters used:
                channels_spec_1, channels_spec_2 ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {(channel_1, micro_time_gates_1)}, {(channel_2, micro_time_gates_2)}
                default_uncertainty_method (whether to try Wohland method SD calculation or directly use bootstrap): {default_uncertainty_method}
                minimum_window_length (shortest allowed window length) [s]: {minimum_window_length}
                n_bootstrap_reps (number of bootstrap replicates to perform) {n_bootstrap_reps}
                tau_min, tau_max (lower and upper bounds of correlation time) [ns]: {tau_min}, {tau_max}
                use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
                use_flcs_bg_corr (use of FLCS-based background subtraction): {use_flcs_bg_corr}
                use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                
                ''' + (f'''Results written to {out_path_full}.csv/png''' if self._write_results else '')
              
            self.write_to_logfile(log_header = '''get_correlation_uncertainty: Calculate correlation function and uncertainty.''',
                                  log_message = log_message,
                                    calling_function = calling_function)

        return lag_times, cc, sd_cc, acr1, acr2
    
    

    #%% Instance methods - TIME TRACE AND BURSTS
    
    def get_trace_time_scale(self, 
                            channels_spec,
                            min_avg_counts = 10.0,
                            min_bin_width = 1E-4,
                            use_tau_diff = False,
                            ext_indices = np.array([]),
                            use_ext_weights = False,
                            use_drift_correction = False,
                            use_flcs_bg_corr = False,
                            use_burst_removal = False,
                            use_mse_filter = False,
                            suppress_logging = False,
                            calling_function = ''
                            ):
        '''
        Automatically estimate a reasonable time scale for time trace 
        construction in the specified channel, based on statistical criteria.
        The idea is that the bin width should:
            1. Contain enough photons on average (min_avg_counts)
            2. Not be stupidly short to avoid inflating the number of data points 
                (min_bin_width). This also serves as a safety net in case the diffusion
                time is misestimated.
            3. If desired, not be shorter than the shortest observed diffusion 
                time (approx. average residence time of molecules in observation 
                volume), which again would be useless oversampling
        The function looks at the three criteria and choses the bin width to fulfil all.

        Parameters
        ----------
        channels_spec: 
            Channel configuration specifier for the correlation operation. 
            See description in self.check_channels_spec() for details.
        min_avg_counts:
            OPTIONAL float or int. Minimum average number of photons per bin. Defaults to 10 photons.
        min_bin_width:
            OPTIONAL float. The absolute minimum time trace resolution in seconds that you want to allow. 
            Defaults to 1E-4 = 100 microseconds.
        use_tau_diff:
            OPTIONAL bool. Whether to use diffusion time analysis to estimate time trace time scale, too. Defaults to False.
        ext_indices :
            OPTIONAL np.array Externally specified indices of photons in the 
            self.photon_data tttr object to use.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. If False (default), ones will be 
            imputed (no specific weighting).
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights. Keep in mind that you would 
            typically use this function as part of the calculation to DETERMINE
            the undrift weights, so think twice about the use here.
        use_flcs_bg_corr :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from FLCS background correction in photon weights.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.
        -------
        time_trace_sampling : 
            float specifying bin width in ns for time trace construction.

        '''
           
        # Make sure that correlation parameters are complete
        if not self._parameters_set:
            self.update_params()

        # input check
        channels_spec_norm = self.check_channels_spec(channels_spec)
        
        # Defaults
        # Minimum time specified
        min_bin_width_ns = min_bin_width * 1E9
        
        # Get (weighted) average count rate
        _, _, weights_select, _ = self.select_photons(channels_spec,
                                                      ext_indices = ext_indices,
                                                      use_ext_weights = use_ext_weights,
                                                      use_burst_removal = use_burst_removal,
                                                      use_drift_correction = use_drift_correction,
                                                      use_flcs_bg_corr = use_flcs_bg_corr,
                                                      use_mse_filter = use_mse_filter,
                                                      suppress_logging = suppress_logging,
                                                      calling_function = 'get_trace_time_scale')
        
        # Get acquistion time from macro_time information, considering possible shortening of acquisition time through corrections
        effective_acquisition_time = self._macro_times[-1]
        
        if use_burst_removal:
            effective_acquisition_time -= self._macro_times_correction_bursts[-1]
            
        if use_mse_filter:
            effective_acquisition_time -= self._macro_times_correction_mse_filter[-1]

        effective_acquisition_time *= self._macro_time_resolution # to ns
        
        acr = weights_select.sum() / effective_acquisition_time # in inverse ns
        
        # From count rate, determine minimum time based on average photon count
        time_for_min_avg_counts = min_avg_counts / acr # in ns
        
        if self._write_log and not suppress_logging:
            log_comment = f'''Parameters used:
                channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm}
                min_avg_counts (criterion 1: minimum average counts per bin): {min_avg_counts}
                min_bin_width (criterion 2: minimum absolute bin width) [s]: {min_bin_width}
                use_tau_diff (whether to use diffusion time estimate as criterion 3): {use_tau_diff}
                ext_indices specified: {ext_indices.shape[0] > 0}
                use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
                use_flcs_bg_corr (use of FLCS-based background subtraction): {use_flcs_bg_corr}
                use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                
                ''' # This function can create different log outputs, this is the stuff that all have in common
            
        if use_tau_diff: 
            # correlation time is wanted as additional criterion
            
            # Get correlation function 
            lag_times, cc, _, _ = self.correlation_apply_filters(channels_spec,
                                                                 channels_spec,
                                                                 use_ext_weights = use_ext_weights, 
                                                                 tau_min = np.array([min_bin_width_ns / 5.]), # Do not waste time looking at stuff far below min_bin_width 
                                                                 use_drift_correction = use_drift_correction,
                                                                 use_flcs_bg_corr = use_flcs_bg_corr,
                                                                 use_burst_removal = use_burst_removal,
                                                                 use_mse_filter = use_mse_filter,
                                                                 suppress_logging = suppress_logging,
                                                                 calling_function = 'get_trace_time_scale')

            lag_times_sec = lag_times / 1E9 # Fitting function operates on seconds time scale
            
            try: 
                
                # First try two-component fit
                diffusion_fit = G_diff_3dim_2comp(lag_times_sec, cc)
                fit_results = diffusion_fit.run_fit()
                short_tau_diff = np.min([fit_results['Tau diffusion 1'], fit_results['Tau diffusion 2']]) * 1E9 # Back to nanoseconds for consistency
                
                # Evaluate short_tau_diff to set time_trace_sampling
                if (short_tau_diff > lag_times[0]) and (short_tau_diff <= 1E8): 
                    # fast diffusion time could be fitted and yields reasonably short binning (<= 100 ms)
                    time_trace_sampling = np.max(np.array([short_tau_diff, time_for_min_avg_counts, min_bin_width_ns]))
                    
                    # Write to log, if desired
                    if self._write_log and not suppress_logging:
                        self.write_to_logfile(log_header = '''get_trace_time_scale: Automatically estimate a reasonable time trace bin width from data.''',
                                              log_message = log_comment + f'''
                                                Used short diffusion time from 2-component fit as criterion 3.
                                                
                                                Fit results:
                                                N: {fit_results['N']}
                                                Diffusion times [s]: {fit_results['Tau diffusion 1']}, {fit_results['Tau diffusion 2']}
                                                Fractions: {fit_results['f1']}, {fit_results['f2']}
                                                Offset: {fit_results['offset']}
                                                
                                                1. time_for_min_avg_counts [ns]: {time_for_min_avg_counts}
                                                2. min_bin_width [ns]: {min_bin_width_ns}
                                                3. tau_diff [ns]: {short_tau_diff}
                                                    
                                                Settled for time_trace_sampling [ns]: {time_trace_sampling}
                                                ''',
                                                calling_function = calling_function)

                else: 
                    # Fit converged, but the value was nonsense: Raise error to trigger single-component fit attempt
                    raise RuntimeError('dummy text, this will never be seen')
                    
            except: # from try: fit_resuts = G_diff_3dim_2comp(...
            
                try: 
                    # Two-component fit failed, try one-component fit
                    diffusion_fit = G_diff_3dim_1comp(lag_times_sec, cc)
                    fit_results = diffusion_fit.run_fit()
                    short_tau_diff = fit_results['Tau diffusion'] * 1E9 # Back to nanoseconds for consistency
                    
                    # Evaluate short_tau_diff to set time_trace_sampling, using same logic as before
                    if (short_tau_diff > lag_times[0]) and (short_tau_diff <= 1E8): 
                        # diffusion time yielded reasonably short binning (<= 100 ms)
                        time_trace_sampling = np.max(np.array([short_tau_diff, time_for_min_avg_counts, min_bin_width_ns]))
                        
                        # Write to log, if desired
                        if self._write_log and not suppress_logging:
                            self.write_to_logfile(log_header = '''get_trace_time_scale: Automatically estimate a reasonable time trace bin width from data.''',
                                                  log_message = log_comment + f'''
                                                    Used diffusion time from 1-component fit as criterion 3.
                                                    
                                                    Fit results:
                                                    N: {fit_results['N']}
                                                    Diffusion time [s]: {fit_results['Tau diffusion']}
                                                    Offset: {fit_results['offset']}
                                                    
                                                    1. time_for_min_avg_counts [ns]: {time_for_min_avg_counts}
                                                    2. min_bin_width [ns]: {min_bin_width_ns}
                                                    3. tau_diff [ns]: {short_tau_diff}
                                                    
                                                    Settled for time_trace_sampling [ns]: {time_trace_sampling}
                                                    ''',
                                                    calling_function = calling_function)

                    else:
                        # Fit converged, but the value was nonsense: Raise error to trigger fall-back solution of using only time_for_min_avg_counts and min_bin_width_ns
                        raise RuntimeError('dummy text, this will never be seen')
                        
                except: # fit_resuts = G_diff_3dim_1comp(...
                    # Neither of the two fit attempts converged - time to give up
                    time_trace_sampling = np.max(np.array([time_for_min_avg_counts, min_bin_width_ns]))
                    
                    # Write to log, if desired
                    if self._write_log and not suppress_logging:
                        self.write_to_logfile(log_header = '''get_trace_time_scale: Automatically estimate a reasonable time trace bin width from data.''',
                                              log_message = log_comment + f'''
                                                Failed to estimate a reasonable diffusion time to use as criterion 3, used only criteria 1 and 2.
                                              
                                                1. time_for_min_avg_counts [ns]: {time_for_min_avg_counts}
                                                2. min_bin_width [ns]: {min_bin_width_ns}

                                                Settled for time_trace_sampling [ns]: {time_trace_sampling}
                                                ''',
                                                calling_function = calling_function)

        else: # if use_tau_diff:
            # No use of correlation time as criterion wanted
            time_trace_sampling = np.max(np.array([time_for_min_avg_counts, min_bin_width_ns]))
                
            # Write to log, if desired
            if self._write_log and not suppress_logging:
                self.write_to_logfile(log_header = '''get_trace_time_scale: Automatically estimate a reasonable time trace bin width from data.''',
                                      log_message = log_comment + '''
                                        Used only count rate and absolute minimum bin width as criteria, ignoring diffusion kinetics.
                                        
                                        1. time_for_min_avg_counts [ns]: {time_for_min_avg_counts}
                                        2. min_bin_width [ns]: {min_bin_width_ns}
                                        
                                        Settled for time_trace_sampling [ns]: {time_trace_sampling}
                                        ''',
                                        calling_function = calling_function)
                                              
        return time_trace_sampling * 1E-9 # Return in seconds


    def get_time_trace(self, 
                        channels_spec,
                        time_trace_sampling,
                        ext_indices = np.array([]),
                        use_ext_weights = False,
                        use_drift_correction = False,
                        use_flcs_bg_corr = False,
                        use_burst_removal = False,
                        use_mse_filter = False,
                        suppress_writing = False,
                        suppress_logging = False,
                        calling_function = ''):
        '''
        Bin photon arrival times into time trace. Pretty straightforward. The
        most notable thing about this function is that it wraps 
        self.select_photons() to select photons based on various criteria, and 
        uses photon weights to allow various floating-point-value corrections.
        
        Parameters
        ----------
        channels_spec: 
            Channel configuration specifier for the correlation operation. 
            See description in self.check_channels_spec() for details.
        time_trace_sampling: 
            Float. Desired time trace resolution in seconds. 
        ext_indices :
            OPTIONAL np.array Externally specified indices of photons in the 
            self.photon_data tttr object to use.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. If False (default), ones will be 
            imputed (no specific weighting).
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights. Keep in mind that you would 
            typically use this function as part of the calculation to DETERMINE
            the undrift weights, so think twice about the use here.
        use_flcs_bg_corr :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from FLCS background correction in photon weights.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        suppress_writing :
            OPTIONAL bool with default False. If true, direct writing of the time trace into
            csv files is suppressed, as these can get ridiculously large and in fact can 
            consume a good part of FCS_Fixer's entire runtime.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        time_trace : 
            np.array with binned trace.
        time_trace_bin_centers :
            Time stamps in seconds that serve as independent variable for time_trace.
        '''
    
        # Input check
        channels_spec_norm = self.check_channels_spec(channels_spec)
                
        # Get photon information
        macro_times_select, _, weights_select, _ = self.select_photons(channels_spec_norm,
                                                                       ext_indices = ext_indices,
                                                                       use_ext_weights = use_ext_weights,
                                                                       use_drift_correction = use_drift_correction,
                                                                       use_flcs_bg_corr = use_flcs_bg_corr,
                                                                       use_burst_removal = use_burst_removal,
                                                                       use_mse_filter = use_mse_filter,
                                                                       suppress_logging = suppress_logging,
                                                                       calling_function = 'get_time_trace')

        macro_times_ns = macro_times_select * self._macro_time_resolution # in ns
    
        # Get acquistion time from macro_time information, considering possible shortening of acquisition time through corrections
        effective_acquisition_time = self._macro_times[-1]
        
        if use_burst_removal:
            effective_acquisition_time -= self._macro_times_correction_bursts[-1]
            
        if use_mse_filter:
            effective_acquisition_time -= self._macro_times_correction_mse_filter[-1]

        effective_acquisition_time *= self._macro_time_resolution # to ns

        # Get actual time trace
        time_trace_bins = np.arange(0, effective_acquisition_time, time_trace_sampling * 1E9, dtype = float)
        time_trace = np.histogram(macro_times_ns,
                                bins = time_trace_bins,
                                density = False,
                                weights = weights_select)[0]
        
        # Return time axis in seconds
        time_trace_bin_centers = (time_trace_bins[:-1] + 0.5 * (time_trace_bins[1] - time_trace_bins[0])) * 1E-9
        
        if self._write_results and not suppress_writing:
            # Write results to 
            out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
            self._out_file_counter += 1
            
            out_path_full += '_Trace_ch' + ''.join([str(element) for element in channels_spec_norm[0]])
            
            # Update name according to applied corrections
            out_path_full += ('_br' if use_burst_removal else '') + \
                             ('_dt' if use_drift_correction else '') + \
                             ('_ar' if use_mse_filter else '') + \
                             ('_w' if use_ext_weights else '') + \
                             ('_bg' if use_flcs_bg_corr else '')


            # Plot and save figure
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(time_trace_bin_centers, time_trace, 'k')
            ax.set_title(f'Fluorescence time trace at bin time {np.round(time_trace_sampling*1E3, 3)} ms') 
            fig.supxlabel('Time [s]')
            fig.supylabel('Counts in bin')
            ax.set_xlim(np.floor(time_trace_bin_centers.min()), np.ceil(time_trace_bin_centers.max()))
            ax.set_ylim(0, np.max(time_trace) * 1.05)

            plt.savefig(out_path_full + '.png', dpi=300)
            plt.close()

            # Create and save spreadsheet
            out_table = pd.DataFrame(data ={'Time[s]': time_trace_bin_centers,
                                            'Counts': np.uint32(time_trace)})
                    
            out_table.to_csv(out_path_full + '.csv', 
                             index = False, 
                             header = self._include_header)

        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''get_time_trace: Bin photons into time trace, applying filters.''',
                                  log_message = f'''Parameters used:
                                    channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm}
                                    time_trace_sampling (time trace bin width) [s]: {time_trace_sampling}
                                    ext_indices specified: {ext_indices.shape[0] > 0}
                                    use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                                    use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
                                    use_flcs_bg_corr (use of FLCS-based background subtraction): {use_flcs_bg_corr}
                                    use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                                    use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                                    
                                    Split {np.round(self._acquisition_time*1E-9, 1)} s long acquisition into {len(time_trace)} bins with an average of {time_trace.mean()} counts per bin.
                                    ''' + (f'''Wrote results to {out_path_full}.csv/png''' if (self._write_results and not suppress_writing) else ''),
                                    calling_function = calling_function)
        
        return time_trace, time_trace_bin_centers


    # Instance methods - BURST REMOVAL
    def get_auto_threshold(self,
                           time_trace,
                           threshold_alpha = 0.02,
                           suppress_logging = False,
                           calling_function = ''):
        '''
        Automatically estimate a threshold photon count for burst detection.
        
        This is done by first approximating the non-burst baseline in an 
        outlier-resistant manner using quantile statistics. This is then 
        approximately converted to parametric (moment) statistics under a 
        Gaussian approximation (which is obviously not accurate, but it's an 
        approximation). Considering that we are interested in detecting bursts, 
        i.e., outliers towards high values, we use the median and 75-percentile
        to approximate the right side of the distribution, ignoring the left side.
        
        Based on this Gaussian parameterization, the threshold is determined 
        such that the value 'burst_threshold' gives the probability of 
        false positive burst detection. This is then Sidak-corrected for 
        multiple comparisons (considering the number of bins) to 
        keep threshold sensitivities somewhat comparable between different binnings.
        np.sqrt(2)*erfcinv(2*threshold_alpha) reads out the threshold of 
        a standard Gaussian at a given alpha, which is then rescaled with sigma 
        and mean(median).
        
        Note that due to the skewed real shape of photon counting histograms,
        the suggested threshold will generally lead to significant
        overthresholding compared to the nominally expected false-positive 
        detection. In practice, we find that threshold_alpha = 0.02 is often a 
        good compromise with decent outlier rejection and harmless false-positive
        rate.
        
        
        Parameters
        ----------
        time_trace : 
            np.array with (binned) time trace to threshold. 
        threshold_alpha :
            Basically a false-positive probability for a time trace bin to be 
            thresholded as burst, given a Gaussian approximation for the baseline
            count rate. Will be Sidak-corrected for the number of bins in the 
            time trace before application.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        threshold_counts :
            int. Absolute number of photons per bin as threshold.
        '''
        
        # Input check for threshold_alpha
        if type (threshold_alpha) != float or (threshold_alpha < 0) or (threshold_alpha > 1):
            raise ValueError('Invalid input for threshold_alpha. Must be float greater than 0 and smaller than 1.')

        # Estimate mean and noise standard deviation of baseline, non-burst, signal.
        [median_signal, upper_1sigma] = np.percentile(time_trace, (50, 84))
        sigma_signal = upper_1sigma - median_signal
        
        # Sidak-correct alpha
        threshold_alpha_Sidak = 1 - ( 1 - threshold_alpha) ** (1 / time_trace.shape[0])
        
        # Use Sidak-corrected alpha to and Gaussian approximation of baseline to get actual threshold
        threshold_counts = int(median_signal + np.sqrt(2) * erfcinv(2 * threshold_alpha_Sidak) * sigma_signal)
        
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''get_auto_threshold: Estimate a reasonable threshold photon count value from data using statistical criteria.''',
                                  log_message = f'''Parameters used:
                                    time_trace: pre-binned photon time trace of length {time_trace.shape[0]} and an average of {time_trace.mean()} counts per bin
                                    threshold_alpha (approx. false-positive thresholding rate to be allowed by threshold): {threshold_alpha}
                                    
                                    Approximated the photon count distribution as a Gaussian with mean {median_signal} and standard deviation {sigma_signal}.
                                    Based on that, settled for a threshold of {threshold_counts}.
                                    ''',
                                    calling_function = calling_function)

        return threshold_counts
            
            
    def threshold_trace(self,
                        time_trace, 
                        threshold_alpha = 0.02,
                        threshold_counts = None,
                        suppress_logging = False,
                        calling_function = ''):
        '''
        Calculate threshold photon count for burst detection, and apply that 
        threshold. 
        This one is quite short, as much of its original content has 
        been stripped off and moved into other functions (used to contain what's now
        get_auto_threshold() and update_photons_from_bursts(). Left it here though 
        for convenience.

        Parameters
        ----------
        time_trace : 
            np.array with (binned) time trace to threshold. 
        threshold_alpha :
            OPTIONAL float. 0 > threshold_alpha > 1. Basically a false-positive probability
            for a time trace bin to be thresholded as burst, given a Gaussian approximation for the baseline
            count rate. Will be Sidak-corrected for the number of bins in the time trace before application.
        threshold_counts :
            OPTIONAL int. Absolute number of photons per bin as threshold. If this is used, the statistical threshold
            tuning based on threshold_alpha is overwritten.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        burst_bins : 
            np.array with bool entries that annotates for each time trace bin 
            whether it is a burst bin or not.
            
        '''
        
        # Define threshold
        if threshold_counts == None:
            # Automatic threshold tuning
            threshold_counts = self.get_auto_threshold(time_trace = time_trace, 
                                                       threshold_alpha = threshold_alpha,
                                                       suppress_logging = suppress_logging,
                                                       calling_function = 'threshold_trace')
            used_auto_threshold = True
            
        elif type(threshold_counts) == int and threshold_counts > 0:
            # Absolute threshold is specified, use that.
            used_auto_threshold = False

            pass
            
        else:
            # Invalid input.
            raise ValueError('Invalid input for threshold_counts. Must be integer > 0 if absolute threshold is to be specified, or exactly 0 (default) for automatic threshold tuning.')
                            
        # Apply threshold to trace
        burst_bins = time_trace > threshold_counts
                            
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''threshold_trace: Apply threshold to a pre-binned time trace to detect bursts.''',
                                  log_message = f'''Parameters used:
                                    time_trace: pre-binned photon time trace of length {time_trace.shape[0]} and an average of {time_trace.mean()} counts per bin
                                    threshold_alpha (approx. false-positive thresholding rate to be allowed in case of auto-thresholding): {threshold_alpha}
                                    -> Auto-thresholding used: {used_auto_threshold}
                                    threshold_counts (actual threshold used): {threshold_counts}
                                    
                                    Thresholded the time trace, identifying {burst_bins.sum()} bins as bursts. 
                                    ''',
                                    calling_function = calling_function)

        return burst_bins
        
        
    def update_photons_from_bursts(self,
                                    burst_bins,
                                    time_trace_sampling,
                                    update_weights = True,
                                    update_macro_times = False,
                                    suppress_logging = False,
                                    calling_function = ''):
        
        '''
        Use a thresholded trace to identify which photons are burst photons.
        Also can update weights to mask our the burst photons and update photon
        times to close gaps from excised bursts, which you'll usually want to do.
        
        Note that burst_bins is a boolean trace of the same length fitting the 
        photon data, but it can come from anywhere. This allows the user to 
        define their own thresholding schemes of summing channels or logically 
        combining them after thresholding individually etc.

        Parameters
        ----------
        burst_bins : 
            np.array of bool with thresholded time trace to further process. 
            Ones indicate "remove", zeros indicate "keep".
        time_trace_sampling: 
            Float. Time trace resolution in seconds.
        update_weights :
            OPTIONAL bool. Whether update the self._weights_burst_removal attribute to mask out burst photons.
        update_macro_times:
            OPTIONAL bool. Whether update the self._macro_times attribute to excise the bursts.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        photon_is_burst : 
            np.array with bool entries that annotates on a photon-by-photon 
            basis which photons are bursts (True: Burst; False: Non-burst). 
            Note that, if update_weights == True, the inverse logical result of 
            this is also written to the property self._weights_burst_removal).
            
        '''
        
        # Start with uncorrected macro_times
        macro_times = self._macro_times.copy()

        # Some setup...
        n_bins = burst_bins.shape[0]
        burst_bin_indices = np.nonzero(burst_bins)[0]
        time_trace_sampling_macro_time_bins = time_trace_sampling * 1E9 / self._macro_time_resolution # in multiples of macro_time_resolution
        
        # Mapping between photons and bins
        first_photons_in_bins = np.zeros((n_bins), dtype = np.uint64)
        # first_photon_in_bin = 0
        current_photon = 0
        bin_edge = 0.
        
        # Iterate over bins to find which is the first photon in the bin
        # We use a while-loop-in-for-loop structure incrementing two time 
        # tags in parallel (current_photon and bin_edge) to avoid redundant comparisons
        for i_bin in range(n_bins):
            
            while macro_times[current_photon] < bin_edge:
                # Stop at the first photon that has a late-enough time tag to fall into this bin
                current_photon += 1
            # Store photon and go to next bin
            first_photons_in_bins[i_bin] = current_photon
            bin_edge += time_trace_sampling_macro_time_bins
            
        # Go back to photons to determine which ones are in burst
        photon_is_burst = np.zeros_like(macro_times, dtype = np.bool8)
        
        # Iterate over burst bins and annotate the photons in that bin
        for i_burst_bin in burst_bin_indices:
            if i_burst_bin < n_bins-1:
                # "Normal" situation
                photon_is_burst[first_photons_in_bins[i_burst_bin]:first_photons_in_bins[i_burst_bin+1]] = True
                
            else:
                # if last bin of trace is burst, we need slightly different handling
                photon_is_burst[first_photons_in_bins[i_burst_bin]:] = True
                                
        if update_weights:
            # Write burst-derived weights to attribute
            self._weights_burst_removal = np.logical_not(photon_is_burst)
            
        else:
            # Store dummy
            self._weights_burst_removal = np.ones_like(macro_times, dtype = np.bool8)
        
        
        if update_macro_times:
                                    
            # Fill macro_times_correction_bursts: For each burst bin, index the photons until the NEXT burst begin and increment their time tag
            if burst_bin_indices.shape[0] > 0:
                # # Found at least one burst
                macro_times_correction_bursts = np.zeros_like(macro_times, dtype = np.uint64)
                                
                # The macro_time correction is performed such that the next 
                # photon that comes after each burst photon - irrespective of whether
                # or not the next photon is a burst photon itself - is placed 
                # at the time point where the first burst photon of the 
                # current burst used to be (in essence, all burst photons and 
                # the next following non-burst photon are all piled up on top 
                # of each other, but that does not matter as we set the weights 
                # of the burst photons to zero - if we do not, this whole 
                # procedure is pretty pointless to be honest.)
                burst_photon_indices = np.nonzero(photon_is_burst)[0]
                
                macro_time_correction = 0.
                
                for i_burst_photon, burst_photon in enumerate(burst_photon_indices[:-1]):
                    macro_time_correction += self._macro_times[burst_photon+1] - self._macro_times[burst_photon]
                    macro_times_correction_bursts[burst_photon+1:burst_photon_indices[i_burst_photon+1]+1] = macro_time_correction
                   
                # Last burst photon: Look more closely: 
                # If the last burst photon is also last photon overall, we do nothing
                # If it is NOT the last photon, we correct all remaining photons
                if burst_photon_indices[-1] < self._n_total_photons - 1:
                    macro_time_correction += self._macro_times[burst_photon+1] - self._macro_times[burst_photon]
                    macro_times_correction_bursts[burst_photon+1:] = macro_time_correction
                                        
                # Store updated macro_times, and convert back to int in the process
                self._macro_times_correction_bursts = np.floor(macro_times_correction_bursts).astype(np.uint64) 
                                    
            else:
                # No bursts - Store dummy
                self._macro_times_correction_bursts = np.zeros_like(macro_times)

        else:
            # No macro_time correction - Store dummy
            self._macro_times_correction_bursts = np.zeros_like(macro_times)
            
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''update_photons_from_bursts: Use thresholded (re-binned) time trace to update photon information.''',
                                  log_message = f'''Parameters used:
                                      burst_bins: thresholded, binary, time trace with {burst_bins.shape[0]} bins of which {burst_bins.sum()} are labelled as burst bins
                                      time_trace_sampling (time trace bin width) [s]: {time_trace_sampling}
                                      update_weights (whether to label the burst photons as photons to exclude from processed by setting their weights to 0): {update_weights}
                                      update_macro_times (whether to update the macro time labled to close gaps in the photon stream created by discarding burst photons): {update_macro_times}
                                      
                                      Out of {photon_is_burst.shape[0]} photons included in burst annotation, {photon_is_burst.sum()} were actually labelled as burst photons.
                                      ''',
                                      calling_function = calling_function)
        
        return photon_is_burst
    
    
    def run_burst_removal(self,
                          time_traces,
                          time_trace_sampling,
                          multi_channel_handling = 'OR',
                          threshold_alpha = 0.02,
                          threshold_counts = None,
                          update_weights = True,
                          update_macro_times = True,
                          suppress_logging = False,
                          calling_function = ''):
        '''
        Convenience function wrapping self.threshold_trace() and 
        self.update_photons_from_bursts() for one-step burst removal, as well 
        as some logic for burst removal handling of multi-channel data. Also 
        auto-creates figured and spreadsheet exports of burst annotation.

        Parameters
        ----------
        time_traces : 
            np.array (1D or 2D) with (binned) time traces to threshold. 1D 
            or 2D with shape[1]==1 for single trace, 2D with traces concatenated
            along axis 1 for multiple traces.
        time_trace_sampling: 
            Float. Time trace resolution in seconds.
        multi_channel_handling : 
            OPTIONAL string. How to handle burst removal across multiple traces.
            Three options:
            'OR' (default): Bin is labelled as burst if it is burst in one 
                or more trace(s). 
            'AND': Bin is labelled as burst if it is burst in every trace.
            'SUM': Traces are added up before burst detection.
        threshold_alpha :
            OPTIONAL float. 0 > threshold_alpha > 1. Basically a false-positive probability
            for a time trace bin to be thresholded as burst, given a Gaussian approximation for the baseline
            count rate. Will be Sidak-corrected for the number of bins in the time trace before application.
        threshold_counts :
            OPTIONAL int. Absolute number of photons per bin as threshold. If this is used, the statistical threshold
            tuning based on threshold_alpha is overwritten.
        update_weights :
            OPTIONAL bool. Whether update the self._weights_burst_removal attribute to mask out burst photons.
        update_macro_times:
            OPTIONAL bool. Whether update the self._macro_times attribute to excise the bursts.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.


        Returns
        -------
        burst_bins : 
            np.array with bool entries that annotates for each time trace bin 
            whether it is a burst bin or not (also in the case of multiple 
            traces, this is a single 1D array with the overall burst annotation 
            result, not the result from individual traces).
        photon_is_burst : 
            np.array with bool entries that annotates on a photon-by-photon 
            basis which photons are bursts (True: Burst; False: Non-burst). 
            Note that, if update_weights == True, the inverse logical result of 
            this is also written to the property self._weights_burst_removal).

        '''
            
        
        
        if time_traces.ndim == 2 and time_traces.shape[1] > 1:
            # Handling of multiple traces at the same time
            
            if not multi_channel_handling in ['OR', 'AND', 'SUM']:
                raise ValueError('Invalid input for multi_channel_handling. Must be "OR", "AND", or "SUM".')
            
            elif multi_channel_handling != "SUM":
                # Implies "OR" or "AND"
            
                burst_bins_channels = np.zeros(time_traces.shape, dtype = np.bool8)
                for i_channel in range(time_traces.shape[1]):
                    burst_bins_channels[:, i_channel] = self.threshold_trace(time_traces[:,i_channel], 
                                                                             threshold_alpha = threshold_alpha,
                                                                             threshold_counts = threshold_counts,
                                                                             suppress_logging = suppress_logging,
                                                                             calling_function = 'run_burst_removal')
                
                if multi_channel_handling == 'OR':
                    # "OR" Handling: Bin is burst is it is burst in any of the channels
                    burst_bins = np.any(burst_bins_channels, axis=1)
                    
                    if self._write_log:
                        log_comment = f'''Joint burst detection in {time_traces.shape[1]} traces via logical OR.'''
                    
                else:
                    # Implies multi_channel_handling == 'AND'
                    # "AND" Handling: Bin is burst if it is burst in all channels
                    burst_bins = np.all(burst_bins_channels, axis=1)
                
                    if self._write_log:
                        log_comment = f'''Joint burst detection in {time_traces.shape[1]} traces via logical AND.'''

            else:
                # Implies "SUM"
                # "SUM" Handling: Bursts are detected in sum of traces
                burst_bins = self.threshold_trace(time_traces.sum(axis=1), 
                                                  threshold_alpha = threshold_alpha,
                                                  threshold_counts = threshold_counts,
                                                  suppress_logging = suppress_logging,
                                                  calling_function = 'run_burst_removal')
                if self._write_log:
                    log_comment = f'''Burst detection on the sum of {time_traces.shape[1]} traces.'''
                
        elif time_traces.ndim == 1 or (time_traces.ndim == 2 and time_traces.shape[1] == 1):
            # Only one trace to deal with
            
            if time_traces.ndim == 2:
                # Single trace, but 2D-formatted - flatten to avoid confusion
                time_traces = time_traces.reshape((time_traces.shape[0],))
                
            burst_bins = self.threshold_trace(time_traces, 
                                              threshold_alpha = threshold_alpha,
                                              threshold_counts = threshold_counts,
                                              suppress_logging = suppress_logging,
                                              calling_function = 'run_burst_removal')
            if self._write_log:
                log_comment = '''Burst detection on a single trace.'''
        
        else:
            raise ValueError('Invalid input for time_traces in run_burst_removal. Must be one-dimensional (one trace) or two-dimensional (one or multiple traces) np.array.')
            
        photon_is_burst = self.update_photons_from_bursts(burst_bins,
                                                          time_trace_sampling,
                                                          update_weights = update_weights,
                                                          update_macro_times = update_macro_times,
                                                          suppress_logging = suppress_logging,
                                                          calling_function = 'run_burst_removal')
        
        # WRite results files , is desired
        if self._write_results:
            # Write results to 
            out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
            self._out_file_counter += 1
            
            out_path_full += '_Burst_annotation' 
            out_path_full += (multi_channel_handling + f'_of_{time_traces.shape[1]}_traces') if time_traces.ndim == 2 and time_traces.shape[1] > 1 else '_single_trace'
            

            # Plot and save figure
            fig, ax = plt.subplots(nrows=1, ncols=1)
            time_trace_bin_centers = np.arange(time_traces.shape[0]) * time_trace_sampling + time_trace_sampling / 2
            
            if time_traces.ndim == 2 and time_traces.shape[1] > 1:
                # Multiple traces
                if multi_channel_handling == "SUM":
                    # Plot sum trace
                    ax.plot(time_trace_bin_centers, 
                            time_traces.sum(axis=1), 
                            marker = '', 
                            linestyle = '-', 
                            color = 'g')
                    # Overlay burst bin markers
                    ax.plot(time_trace_bin_centers[burst_bins], 
                            time_traces.sum(axis=1)[burst_bins], 
                            marker = 'd', 
                            linestyle = '', 
                            color = 'g')     
                    ax.set_title(f'SUM of {time_traces.shape[1]} traces with burst annotation') 

                    # Create spreadsheet
                    out_table = pd.DataFrame(data ={'Time[s]': time_trace_bin_centers,
                                                    'Counts': np.uint32(time_traces.sum(axis=1)),
                                                    'Is_burst': np.uint8(burst_bins)})

                else:
                    # Implies OR or AND
                    # Multiple traces, plot individually cycling through colors
                    prop_cycle = plt.rcParams['axes.prop_cycle']
                    colors = cycle(prop_cycle.by_key()['color'])
                    
                    # Create spreadsheet
                    out_table = pd.DataFrame(data ={'Time[s]': time_trace_bin_centers,
                                                    f'Is_burst_{multi_channel_handling}': np.uint8(burst_bins)})

                    for i_channel in range (time_traces.shape[1]):
                        iter_color = next(colors)
                        # Trace itself
                        ax.plot(time_trace_bin_centers, 
                                time_traces[:,i_channel], 
                                marker = '', 
                                linestyle = '-', 
                                alpha = 0.7,
                                color = iter_color)
                        # Overlay burst bin markers
                        ax.plot(time_trace_bin_centers[burst_bins_channels[:, i_channel]], 
                                time_traces[burst_bins_channels[:, i_channel],i_channel], 
                                marker = 'd', 
                                linestyle = '', 
                                color = iter_color)
                        ax.set_title(f'{time_traces.shape[1]} traces with individual burst annotation') 
                        
                        # Add to spreadsheet
                        out_table[f'Counts_{i_channel}'] = time_traces[:,i_channel]
                        out_table[f'Is_burst_{i_channel}'] = np.uint8(burst_bins_channels[:, i_channel])

            else:
                # Single trace
                # Plot trace
                ax.plot(time_trace_bin_centers, 
                        time_traces, 
                        marker = '', 
                        linestyle = '-', 
                        color = 'k')
                # Overlay burst bin markers
                ax.plot(time_trace_bin_centers[burst_bins], 
                        time_traces[burst_bins], 
                        marker = 'd', 
                        linestyle = '', 
                        color = 'k')     
                ax.set_title('Single trace with burst annotation') 

                # Create spreadsheet
                out_table = pd.DataFrame(data ={'Time[s]': time_trace_bin_centers,
                                                'Counts': np.uint32(time_traces),
                                                'Is_burst': np.uint8(burst_bins)})

            # Same irrespective of number of traces and multi_channel_handling
            fig.supxlabel('Time [s]')
            fig.supylabel('Counts in bin')
            ax.set_xlim(np.floor(time_trace_bin_centers.min()), np.ceil(time_trace_bin_centers.max()))
            ax.set_ylim(0, np.max(time_traces) * 1.05)

            plt.savefig(out_path_full + '.png', dpi=300)
            plt.close()

            out_table.to_csv(out_path_full + '.csv', 
                             index = False, 
                             header = self._include_header)

        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''run_burst_removal: Use one or multiple time traces for burst removal.''',
                                  log_message = f'''Parameters used:
                                      time_trace: {time_traces.shape[1] if time_traces.ndim == 2 else 1} pre-binned photon time traces of length {time_traces.shape[0]} and averages of {time_traces.mean(axis=0)} counts per bin
                                      time_trace_sampling (time trace bin width) [s]: {time_trace_sampling}
                                      threshold_alpha (approx. false-positive thresholding rate to be allowed in case of auto-thresholding): {threshold_alpha}
                                      threshold_counts (actual threshold used): {threshold_counts}
                                      update_weights (whether to label the burst photons as photons to exclude from processed by setting their weights to 0): {update_weights}
                                      update_macro_times (whether to update the macro time labled to close gaps in the photon stream created by discarding burst photons): {update_macro_times}
                                      
                                      ''' + log_comment + f'''
                                      Overall detected {burst_bins.sum()} burst bins in {time_traces.shape[0]} bins.                                      
                                      Out of {photon_is_burst.shape[0]} photons included in burst annotation, {photon_is_burst.sum()} were actually labelled as burst photons.
                                      ''' + (f'''Wrote results to {out_path_full}.csv/png''' if self._write_results else ''),
                                      calling_function = calling_function)
                                      
        return burst_bins, photon_is_burst
        

#%% Instance methods - BLEACHING/DRIFT CORRECTION
    
    def get_auto_undrift_order(self,
                               time_trace_bin_centers,
                               time_trace_counts,
                               max_undrift_order):
        '''
        Iterate through polynomial degrees, repeatedly fitting a dataset and 
        use F-test statistics to determine a polynomial degree that fits
        the data well.

        Parameters
        ----------
        time_trace_bin_centers : 
            np.array with independent variable for fits.
        time_trace_counts : 
            np.array with dependent variable for fits. Will also be used for
            fit weighting under Poisson statistics assumption.
        max_undrift_order : 
            Int. Highest order up to which the polynomial in increased before 
            giving up and considering the dataset non-converging.

        Returns
        -------
        undrift_order : 
            Int. Ideal polynomial degree to use for fitting the dataset in 
            question.

        '''
        undrift_order = 1
        red_chi_sq = np.zeros((max_undrift_order,), dtype=float) # Reduced chi-square
        significant_improvement= np.zeros((max_undrift_order,), dtype=bool)
        
        while undrift_order <= max_undrift_order: 
            # We increment order until it is enough, but not beyond max_undrift_order
            # Fit polynomial
            # We use a custom class instead of np.polynomial.polynomial.polyfit() 
            # because you cannot constrain parameters in polyfit. This hardly 
            # ever matters, but rarely, on awkward data, polyfit() can fit 
            # the 0-order term as < 0. This is not only unphysical for a 
            # photon count time trace, it also blows up the math in the 
            # line where weights_undrift_select is calculated, and corrupts 
            # everything that comes afterwards. Note that this ONLY concerns
            # the 0-order term, the higher orders are freely fitted.
            
            # While iterating over polynomial orders, we only need the 
            # reduced chi-square and do not yet care about the actual fit parameters
            _, red_chi_sq_iter = Polynomial_fit(time_trace_bin_centers, 
                                           time_trace_counts, 
                                           undrift_order).run_fit()
            
            red_chi_sq[undrift_order-1] = red_chi_sq_iter
            
            if undrift_order>1:
                # dfn = 1 # F-distribution numerator degrees of freedom is always 1 as we increment by one parameter every time, so we do not need to explicitly mention it in calculation
                dfd = time_trace_counts.shape[0] - undrift_order # F-distribution degrees of freedom in denominator
                F_value = (red_chi_sq[undrift_order-2] - red_chi_sq[undrift_order-1]) / red_chi_sq[undrift_order-1] * dfd
                significant_improvement[undrift_order-1] = f_dist.sf(F_value, 1, dfd) < 0.05
                
                if (significant_improvement[undrift_order-1] or significant_improvement[undrift_order-2]) and undrift_order != max_undrift_order:
                    # Makes sense to look at another iteration
                    # We look at two consecutive improvements as we swap between adding even and odd orders
                    undrift_order += 1
                    
                elif undrift_order == max_undrift_order:
                    # We reached order max_undrift_order. Time to stop, but let's see which undrift_order we leave...
                    
                    if significant_improvement[undrift_order-1] and significant_improvement[undrift_order-2]:
                        # No convergence at all. Abort with undrift_order = max_undrift_order
                        break
                    
                    elif not significant_improvement[undrift_order-2]:
                        # Looks like max_undrift_order-2 was enough although max_undrift_order-1 then did bring a further improvement, weird and certainly rare case, but whatever.
                        undrift_order -= 2
                        break
                    
                    else: # Implies: significant_improvement[undrift_order-2] and not significant_improvement[undrift_order-1]
                        # Looks like max_undrift_order-1 was enough
                        undrift_order -= 1
                        break
                else:
                    # Twice in a row no improvement: That's it for sure.
                    # As we used 2 consecutive non-improvements as cutoff, we go back two steps
                    undrift_order -= 2
                    break
            else:
                undrift_order += 1 # This one is only needed to break out of infinite loop on undrift_order == 1
                
        return undrift_order
    

    def polynomial_undrifting_rss(self,
                                  time_trace,  
                                  time_trace_bin_centers, 
                                  channels_spec,
                                  undrift_order = None,
                                  max_undrift_order = 10,
                                  update_undrift_weights = True,
                                  ext_indices = np.array([]),
                                  use_ext_weights = False,
                                  use_flcs_bg_corr = False,
                                  use_burst_removal = False,
                                  use_mse_filter = False,
                                  suppress_logging = False,
                                  calling_function = ''):
        '''
        Fit a polynomial of either user-defined or automatically chosen order
        to determine bleaching/drift correction of photon time trace.
        
        A polynomial fit is used to describe the coarse trend in the data, and 
        then this trend is corrected based on eq. 4 in Ries, Chiantia and 
        Schwille Biophys J 2009, DOI: 10.1016/j.bpj.2008.12.3888. 
        
        The order of the polynomial fit can be explicitly specified by the user,
        or optimized automatically. If automatic order tuning is chosen, the 
        software starts with a zero-order polynomial (= no correction at all) and
        increments the order, inspecting statistical goodness-of-fit improvements.
        The order is considered optimized if two further increments in the order 
        did not significantly change the goodness of fit. (Two increments rather 
        than only one as the software alternates between even and odd orders, 
        which may behave differently.)
        
        Note that while this will not very often be meaningful, it is in principle 
        possible to use a time trace constructed from one photon selection 
        correct a totally different selection of photons. For that to work mathematically,
        it is not even necessary for the channels to have similar count rates.
        One way how this could be useful is to apply undrifting to a sum trace 
        over multiple channels with high signal-to-noise ratio, only to correct
        only a single, dim, channel. 
        
        Parameters
        ----------
        time_trace : 
            np.array with binned trace.
        time_trace_bin_centers :
            Time stamps in seconds (!) that serve as independent variable for 
            time_trace. Take care that this should be bin centers as returned by
            self.get_time_trace(), not bin edges.
        channels_spec: 
            Channel configuration specifier for the which photons to correct. 
            Note that really only the photons indicated by this thing together 
            with other filters will be included in the correction.
            See description in self.check_channels_spec() for details.
        undrift_order :
            OPTIONAL int or None (default, means "choose automatically"). Order
            of the polynomial fit to use for undrifting.
        max_undrift_order :
            OPTIONAL int. If undrift_order == None, the order is automatically 
            determined. This is the maximum order that the user allows. Default 10.
        update_undrift_weights :
            OPTIONAL bool with default True. Whether to write the results of the
            undrifting into the self._weights_undrift attribute. Normally one 
            will want to use that, yes, as that is kind of the whole point of 
            this function.
        ext_indices :
            OPTIONAL np.array Externally specified indices of photons in the 
            self.photon_data tttr object to use. Part of the decision which 
            photons to correct.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. If False (default), ones will be 
            imputed (no specific weighting). 
        use_flcs_bg_corr :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from FLCS background correction in photon weights.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        poly_params :
            np.array with the polynomial fit parameters found by np.polynomial.polyfit()
        time_trace_poly : 
            np.array fitted model of bleaching/drift trend.
        time_trace_undrift :
            Recalculated time trace after applying the just-found correction.
        '''

        # Input check
        channels_spec_norm = self.check_channels_spec(channels_spec)
        
        # From seconds to ns
        time_trace_bin_centers *= 1E9
            
        if undrift_order ==  None:
            # Automatically tune order
            
            undrift_order = self.get_auto_undrift_order(time_trace_bin_centers,
                                                        time_trace,
                                                        max_undrift_order)
                    
            # For logging of cases with auto-tuning of order
            log_comment = '''Auto-tuned undrift fit order based on residual sum of squares.'''
            
        else: # No auto-tuning of undrifting fit order
            log_comment = '''Using user-supplied undrift fit order.'''

        # Whatever we did before, now we have decided undrift_order.
        # Fit polynomial
        poly_params, _ = Polynomial_fit(time_trace_bin_centers, 
                                        time_trace, 
                                        undrift_order).run_fit()

        # Get photon information
        macro_times_select, _, weights_select, indices_select = self.select_photons(channels_spec_norm, 
                                                                                    ext_indices = ext_indices,
                                                                                    use_ext_weights = use_ext_weights, 
                                                                                    use_burst_removal = use_burst_removal,
                                                                                    use_flcs_bg_corr = use_flcs_bg_corr,
                                                                                    use_mse_filter = use_mse_filter,
                                                                                    suppress_logging = suppress_logging,
                                                                                    calling_function = 'polynomial_undrifting_rss')
        
        macro_times_select_ns = macro_times_select * self._macro_time_resolution
        
        # Get correction from fitted polynomial
        # Polynomial_fit() custom class uses the same expression as 
        # np.polynomial.polynomial.polyval(), we can simply use that here
        poly_values = np.polynomial.polynomial.polyval(macro_times_select_ns, poly_params)
        poly_zero = np.polynomial.polynomial.polyval(0., poly_params)
        
        weights_undrift_select = 1. / np.sqrt(poly_values / poly_zero) + (1 - np.sqrt(poly_values / poly_zero))
        weights_undrift_select[np.isnan(weights_undrift_select)] = 0. # can still happen, poly_values can sometimes be negative although admittedly that does not make sense physically
        
        # Set to property if specified
        if update_undrift_weights:
            self._weights_undrift[indices_select] = weights_undrift_select
                
        # Calculate and return polynomial time trace model itself
        time_trace_poly = np.polynomial.polynomial.polyval(time_trace_bin_centers, poly_params)

        # Reconstruct time bins for trace generation
        time_trace_sampling = time_trace_bin_centers[1] - time_trace_bin_centers[0]
        time_trace_bins = np.append(time_trace_bin_centers, time_trace_bin_centers[-1] + time_trace_sampling)
        time_trace_bins -= 0.5 * time_trace_sampling
        
        time_trace_undrift, _ = np.histogram(macro_times_select_ns, 
                                             bins = time_trace_bins, 
                                             weights = weights_undrift_select * weights_select, 
                                             density=False)
        time_trace_undrift = time_trace_undrift.astype(np.float64)
        
        if self._write_results:
            # Write results to 
            out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
            self._out_file_counter += 1
            
            out_path_full += '_Trace_ch' + ''.join([str(element) for element in channels_spec_norm[0]])
            
            # Update name according to applied corrections
            out_path_full += ('_br' if use_burst_removal else '') + \
                             ('_ar' if use_mse_filter else '') + \
                             ('_w' if use_ext_weights else '') + \
                             ('_bg' if use_flcs_bg_corr else '') + \
                             '_undrift_fit'

            # Plot and save figure
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(time_trace_bin_centers * 1E-9, time_trace, 'g', label = 'Raw')
            ax.plot(time_trace_bin_centers * 1E-9, time_trace_undrift, 'm', alpha = 0.5, label = 'Correction')
            ax.plot(time_trace_bin_centers * 1E-9, time_trace_poly, 'k', label = 'Fit')
            ax.legend()
            ax.set_title(f'Time trace undrifting polynomial fit order {undrift_order}') 
            fig.supxlabel('Time [s]')
            fig.supylabel('Counts in bin')
            ax.set_xlim(np.floor(time_trace_bin_centers.min()*1E-9), np.ceil(time_trace_bin_centers.max()*1E-9))
            ax.set_ylim(0, np.max([np.max(time_trace), np.max(time_trace_undrift)]) * 1.05)

            plt.savefig(out_path_full + '.png', dpi=300)
            plt.close()

            # Create and save spreadsheet
            out_table = pd.DataFrame(data ={'Time[s]': time_trace_bin_centers*1E-9,
                                            'Counts_Raw': np.uint32(time_trace),
                                            'Fit': time_trace_poly,
                                            'Counts_Corrected': time_trace_undrift})
                    
            out_table.to_csv(out_path_full + '.csv', 
                             index = False, 
                             header = self._include_header)


        if self._write_log and not suppress_logging:
            undrift_expr_string = 'b(t[ns]) = ' + str(poly_params[0]) + ''.join([' + (' + str(poly_params[order]) + ')*t^' + str(order) for order in range(1, undrift_order+1)])
            
            self.write_to_logfile(log_header = '''polynomial_undrifting_rss: Fit time trace with a polynomial model to describe bleaching/drift trend, and correct it.''',
                                  log_message = f'''Parameters used:
                                time_trace: pre-binned photon time trace of length {time_trace.shape[0]} and an average of {time_trace.mean()} counts per bin
                                time_trace_bin_centers: bin centers [ns] for time trace, binwidth is {time_trace_bin_centers[1] - time_trace_bin_centers[0]} ns.
                                channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm}
                                undrift_order (polynomial fit order used to describe the beaching/drift trend): {undrift_order}
                                -> '''+log_comment+f'''
                                max_undrift_order (in case of auto-tuning of undrift-order, what was the maximum allowed order): {max_undrift_order}
                                update_undrift_weights (whether to write the results of undrifting into the photon weights for further processing): {update_undrift_weights}
                                ext_indices specified: {ext_indices.shape[0] > 0}
                                use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                                use_flcs_bg_corr (use of FLCS-based background subtraction): {use_flcs_bg_corr}
                                use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                                use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                            
                                Corrected {indices_select.shape[0]} photons with bleaching/drift trend model model b(t[ns]):                                      
                                '''+undrift_expr_string+'''
                                ''' + (f'''Wrote results to {out_path_full}.csv/png''' if self._write_results else ''),
                                calling_function = calling_function)



        return poly_params, time_trace_poly, time_trace_undrift
        

    
    #%% INSTANCE METHODS - SEGMENTS WITH ANOMALOUS CORRELAITON FUNCTION    
    
    def cc_mse_filter_get_mse(self,
                            segment_ccs,
                            ignore_amplitude_fluctuations = True,
                            suppress_logging = False,
                            calling_function = ''):
        '''
        Run an iterative comparison of correlation functions based on sum-of-
        mean-square-error statistics. In each iteration, the curve that has the 
        strongest mismatch to the average of all others is discarded, and this
        iterative removal is continued until only 2 are left (at which point
        this obviously no longer works).

        Parameters
        ----------
        segment_ccs: 
            np.array (2D). Correlation functions of all time segments to compare. 
            Axis 0 is iteration over lag times, axis 1 of length equal number 
            of segments into which time axis was segmented.
        ignore_amplitude_fluctuations : 
            Optional bool with default True. If True, in each comparison, the 
            correlation curve whose MSE is calculated is linearly scaled to
            minimize the MSE from the average of the other curves. The idea is 
            that this way, the effect of amplitude changes is suppressed and 
            instead the emphasis is on curve shape chamges.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        mean_sq_error : 
            np.array (2D). Records each curve's MSE at each iteration. Axis 0 is
            iteration over curves, axis 1 over removal iterations. Comparisons
            that were not made (remember, in each iteration one curve is 
            removed) are left with NaN values.
        mse_at_removal : 
            np.array (1D). For each curve, records the MSE value that was found 
            for this curve in the iteration in which it was discarded. For the two 
            curves that are never discarded, the value is NaN.
        kept_until_iteration : 
            np.array (1D). For each curve, records up to which removal iteration
            this curve was kept in the process. The higher the number, the more
            consistant the curve was with the rest. Note that as the process stops
            with 2 curves left, the highest value in kept_until_iteration occurs 
            twice in the array.

        '''
        keep_indices = np.arange(segment_ccs.shape[1])
        kept_until_iteration = np.zeros((segment_ccs.shape[1]), dtype = np.uint16)
        
        mean_sq_error = np.zeros((segment_ccs.shape[1], segment_ccs.shape[1]-2), dtype = np.float32)
        mean_sq_error[:] = np.nan
        mse_at_removal = np.zeros((segment_ccs.shape[1]), dtype = np.float32)
        mse_at_removal[:] = np.nan
        
        # Loop to determine which of the segments has the least agreement with the others 
        for iteration in range(segment_ccs.shape[1]-2):
            
            # Loop to compare each segment to all remaining ones
            for segment_index in keep_indices:
                # Get mean of all still-kept segment CCs except this one
                mean_other_segments = np.mean(segment_ccs[:, keep_indices[keep_indices!=segment_index]], 1)
                
                if ignore_amplitude_fluctuations:
                    # Scale segment to mean of other segments to emphasize cuve shape deviations over curve amplitude deviations
                    segment_cc = lin_scaling_of_data(segment_ccs[:,segment_index], mean_other_segments)
                else:
                    # No scaling, also letting amplitude fluctuations influence the result
                    segment_cc = segment_ccs[:,segment_index]
                    
                # Variance-normalized mean-squares deviation as the actual agreement/discrepancy parameter
                mean_sq_error[segment_index, iteration] = np.nanmean((segment_cc - mean_other_segments)**2)
                
            # Who has to leave, who can stay?
            discard_indx = np.nanargmax(mean_sq_error[:, iteration])
            keep_indices = keep_indices[keep_indices != discard_indx]
            
            # Place/update markers
            mse_at_removal[discard_indx] = mean_sq_error[discard_indx, iteration] 
            kept_until_iteration[keep_indices] += 1
        
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''cc_mse_filter_get_mse: Iteratively compare correlation functions to flag potential outliers through mean-square error statistics.''',
                                  log_message = f'''Parameters used:
                                      segment_ccs: Set of {segment_ccs.shape[1]} correlation functions to compare, each containing {segment_ccs.shape[0]} data points
                                      ignore_amplitude_fluctuations (whether to linearly scale data to emphasise correlation function shape anomalies over amplitude fluctuations): {ignore_amplitude_fluctuations}
                                      
                                      Found mean-square error values varying over a factor of {np.round(np.nanmax(mse_at_removal) / np.nanmin(mse_at_removal), 1)}. 
                                      ''',
                                      calling_function = calling_function)
                                      
        return mean_sq_error, mse_at_removal, kept_until_iteration
    
    
    def run_mse_filter(self,
                       channels_spec_1,
                       channels_spec_2,
                       minimum_window_length = [],
                       mse_fold_threshold = 2.5,
                       ignore_amplitude_fluctuations = True,
                       update_macro_times = True,
                       update_weights = True,
                       tau_max = None,
                       tau_min = 1E5, # 100 us as default, to cut off excessive noise
                       use_ext_weights = False, 
                       use_drift_correction = False, 
                       use_flcs_bg_corr = False,
                       use_burst_removal = False,
                       suppress_logging = False,
                       calling_function = ''):
        '''
        Split the measurement into several segment correlation functions, and 
        compare them based on the mean square error between them. Based on a 
        threshold of fold change over the lowest encountered mean-square error,
        discard segments that seems to be behave anomalously. 
        
        This is based on Ries, ..., Schwille Optics Express 2010, DOI: 10.1364/OE.18.011073

        Parameters
        ----------
        channels_spec_1, channels_spec_2 : 
            Channel configuration specifier for the photons to include in 
            correlation function calculations. See description in 
            self.check_channels_spec() for details.
        minimum_window_length:
            OPTIONAL float > tau_max. Specifies minimum width for the windows 
            in which to separate the trace. In seconds.
        mse_fold_threshold : 
            OPTIONAL float with default 2.5. Discarding of segmentas is based on
            mean-square-error statistics. This parameter specifies a fold change
            of MSE over the smallest MSE with which any curve was removed during
            comparison for deciding from which point on to actually discard 
            segments.
        ignore_amplitude_fluctuations : 
            OPTIONAL bool with default True. If True, in each comparison, the 
            correlation curve whose MSE is calculated is linearly scaled to
            minimize the MSE from the average of the other curves. The idea is 
            that this way, the effect of amplitude changes is suppressed and 
            instead the emphasis is on curve shape chamges.
        update_macro_times : 
            OPTIONAL bool with default True. Whether to write the results of 
            segment removal into the self._macro_times_correction_mse_filter
            property to update time tags to close the gaps from removal of 
            segments.
        update_weights : 
            OPTIONAL bool with default True. Whether to write the results of 
            segment removal into the self._weights_mse_filter
            property to mask out photons from time segments labelled as 
            anomalous.
        tau_min, tau_max:
            OPTIONAL specifiers for minimum and maximum lag time to correlate. 
            float in ns. If empty or not specified at all, will be replaced by 
            global defaults.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. OPTIONAL. If False (default), ones will be 
            imputed (no specific weighting).
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights.
        use_flcs_bg_corr :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from FLCS background correction in photon weights.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.


        Returns
        -------
        mean_sq_error : 
            np.array (2D). Records each curve's MSE at each iteration. Axis 0 is
            iteration over curves, axis 1 over removal iterations. Comparisons
            that were not made (remember, in each iteration one curve is 
            removed) are left with NaN values.
        good_segments
            np.array (1D). Indices of segments classified as "good" by the filter.

        '''
        
        # Make sure that correlation parameters are complete
        if not self._parameters_set:
            self.update_params()
        
        # Input check
        tau_min, tau_max = self.check_tau_min_max(tau_min, tau_max)
        channels_spec_norm_ch1 = self.check_channels_spec(channels_spec_1)    
        channels_spec_norm_ch2 = self.check_channels_spec(channels_spec_2)    
        
        # Get acquistion time from macro_time information, considering possible shortening of acquisition time through corrections
        effective_acquisition_time = self._macro_times[-1]
        
        if use_burst_removal:
            effective_acquisition_time -= self._macro_times_correction_bursts[-1]
            
        effective_acquisition_time *= self._macro_time_resolution # to ns

        
        if type(minimum_window_length) in [list, np.ndarray] and len(minimum_window_length) == 0:
            # Auto-choose
            minimum_window_length = np.max([effective_acquisition_time/10., 5.*tau_max]).astype(np.float64) * 1E-9 # to seconds for now
            if self._write_log and not suppress_logging:
                log_comment_minimum_window_length = f'''minimum_window_length (shortest allowed window length) [s]: Automatically chosen as {minimum_window_length}'''

        elif type(minimum_window_length) in [float, np.float32, np.float64] and minimum_window_length * 1E9 > tau_max:
            # Valid user input
            if self._write_log and not suppress_logging:
                log_comment_minimum_window_length = f'''minimum_window_length (shortest allowed window length) [s]: Chosen by user as {minimum_window_length}'''
            minimum_window_length = np.float64(minimum_window_length)
            
        else:
            raise ValueError('minimum_window_length must be float or empty list. minimum_window_length must be > tau_max. Keep in mind that this function interprets the input for tau_max as being in nanoseconds, the input for minimum_window_length as in seconds!')
            
        if self._write_log and not suppress_logging:
            log_comment = f'''Parameters_used:
                channels_spec_1, channels_spec_2 ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm_ch1}, {channels_spec_norm_ch2}
                '''+log_comment_minimum_window_length+f'''
                mse_fold_threshold (mean-square error fold-change threshold for discarding segments): {mse_fold_threshold}
                ignore_amplitude_fluctuations (whether to linearly scale data to emphasise correlation function shape anomalies over amplitude fluctuations): {ignore_amplitude_fluctuations}
                update_macro_times (whether to update the macro time labled to close gaps in the photon stream created by discarding burst photons): {update_macro_times}
                update_weights (whether to label the burst photons as photons to exclude from processed by setting their weights to 0): {update_weights}
                tau_min, tau_max (lower and upper bounds of correlation time) [ns]: {tau_min}, {tau_max}
                use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
                use_flcs_bg_corr (use of FLCS-based background subtraction): {use_flcs_bg_corr}
                use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                ''' # for logging, this function can create different log outputs, this is (except for the minimum_window_length stuff) the stuff that's always the same

        # At least 3 windows would be nice for Ries method anomaly filter. 
        # If that is not fulfilled, there's actually nothing we can do, return dummy.
        if effective_acquisition_time * 1E-9 / minimum_window_length < 3: 
            
            # Write to log, if desired
            if self._write_log and not suppress_logging:
                self.write_to_logfile(log_header = '''run_mse_filter: Segment photon stream into time windows, correlate them, and remove outliers with anomalous correlation function.''',
                                      log_message = log_comment + '''
                                        Could not fit enough segments of specified window size into acquisition time. Skipped.
                                        ''',
                                      calling_function = calling_function)
                                      
            return np.array([]), np.array([])     
        
        else:
            # We get enough windows.
            lag_times, segment_ccs, usable_segments, start_stop = self.get_segment_ccs(channels_spec_norm_ch1,
                                                                                       channels_spec_norm_ch2,
                                                                                       minimum_window_length,
                                                                                       tau_min = tau_min,
                                                                                       tau_max = tau_max,
                                                                                       use_ext_weights = use_ext_weights,
                                                                                       use_drift_correction = use_drift_correction,
                                                                                       use_flcs_bg_corr = use_flcs_bg_corr,
                                                                                       use_burst_removal = use_burst_removal,
                                                                                       use_mse_filter = False,
                                                                                       suppress_logging = suppress_logging,
                                                                                       calling_function = 'run_mse_filter')
            # Crop out the unusable ones
            segment_ccs = segment_ccs[:,usable_segments]
            
            # Run comparison to score how consistent segment ccs are with other segments and keep iteratively discarding the worst one, until only 2 are left.
            mean_sq_error, mse_at_removal, kept_until_iteration = self.cc_mse_filter_get_mse(segment_ccs,
                                                                                             ignore_amplitude_fluctuations = ignore_amplitude_fluctuations,
                                                                                             suppress_logging = suppress_logging,
                                                                                             calling_function = 'run_mse_filter')
                                    
                
            # Threshold acceptable cases: Within mse_fold_threshold * minimum MSE
            # Actually, we look for the last iteration in which we found an
            # inacceptable mean_sq_error and discard everything from iterations up to that point
            bad_segments = np.nonzero(mse_at_removal > np.nanmin(mse_at_removal) * mse_fold_threshold)[0]

            if bad_segments.shape[0] > 0: 
                # At least one "bad segment": Sort what to keep and what to discard
                
                # Which is the last iteration up to which we had to discard?
                last_bad_iteration = np.max(kept_until_iteration[bad_segments])
            
                # The "good segments" are all that are kept longer than the last bad segment
                good_segments = usable_segments[kept_until_iteration > last_bad_iteration]
                
            else:
                # Nothing discarded
                good_segments = usable_segments

            if update_macro_times or update_weights:
                # If we update at least one of the arrtibutes, we continue, otherwise
                # we can just return the mean_sq_error matrix

                # Now we crop the data to include the good_segments only
                good_start_stop = start_stop[good_segments,:]
                
                # Merge successive good segments into a single one to be more clear
                # about where bad segments start
                good_start_stop_sort = self.sort_start_stop(good_start_stop)
                        
                if good_start_stop_sort.shape[0] > 1 or good_start_stop_sort[0,0] > 0 or good_start_stop_sort[-1,1] < self._n_total_photons - 1:
                    # If at least one of these criteria is false, we did discard something.
                                    
                    if update_macro_times:
                        
                        # Which macro_times we update depends on whether we had already excised bursts...
                        macro_times = self._macro_times.copy()
                        
                        # Silenced currently as this correction is broken. Will be debugged later, but it's not too important
                        if use_burst_removal:
                            macro_times -= self._macro_times_correction_bursts
                        
                        # Correct macro_times for excised gaps
                        macro_times_correction = np.zeros_like(macro_times)
                        
                        # Get gap sizes: 
                        # The last photon of a "bad segment" is one before the first photon of the following "good segment" 
                        # The first photon of a "bad segment" is one after the last photon of the preceding "good segment"
                        # The gap size (in macro time units) is the difference of their time tags
                        if good_start_stop_sort[0,0] > 0:
                            # We discarded the first segment
                            
                            # First correct segment 0
                            macro_times_correction[good_start_stop_sort[0,0]:] += macro_times[good_start_stop_sort[0,0]-1]
                            
                            if good_start_stop_sort.shape[0] > 1:
                                # Get and correct gap sizes for following gaps, if any
                                gap_sizes = macro_times[good_start_stop_sort[1:,0]-1] - macro_times[good_start_stop_sort[:-1,1]]
                                
                                # Sum up gap sizes and assign to photon-wise correction
                                # We corrects photon starting with the first photon of the "good" segment following each "bad" segment
                                for gap_indx, gap_size in enumerate(gap_sizes):
                                    macro_times_correction[good_start_stop_sort[gap_indx+1,0]:] += gap_size
                            
                        else:
                            # The first segment was kept
                            
                            if good_start_stop_sort.shape[0] > 1:
                                # We have at least 2 good segments to effeticely join together

                                gap_sizes = macro_times[good_start_stop_sort[1:,0]-1] - macro_times[good_start_stop_sort[:-1,1]]
                                                        
                                # Sum up gap sizes and assign to photon-wise correction
                                # We find the correct photon as being after each "good" segment
                                for gap_indx, gap_size in enumerate(gap_sizes):
                                    macro_times_correction[good_start_stop_sort[gap_indx+1,0]:] += gap_size
                            
                            else:
                                # Only one good segment
                                macro_times_correction[good_start_stop_sort[0,0]:] += macro_times[good_start_stop_sort[0,0]-1]


                        # Assign shifted macro times to property
                        self._macro_times_correction_mse_filter = np.uint64(np.floor(macro_times_correction))
                        
                    if update_weights:
                        # Create weights array with zeros for photons to be discarded
                        # and ones for photons to be used.
                        weights_anomalous_segments = np.zeros((self._n_total_photons,), dtype = np.bool8)
                        for start, stop in good_start_stop_sort:
                            weights_anomalous_segments[start:stop] = True
                        
                        # Assign to property
                        self._weights_mse_filter = weights_anomalous_segments
                        
                        if self._write_log and not suppress_logging:
                            # If we do this, we add a bit of info to the log
                            log_comment += f'''
                            Out of {self._n_total_photons} photons, {weights_anomalous_segments.shape[0] - weights_anomalous_segments.sum()} were discarded.
                            '''
            
            
            if self._write_results:
                # Write results to csv and png output
                out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
                self._out_file_counter += 1
                
                
                channel_1, micro_time_gates_1 = channels_spec_norm_ch1
                channel_2, micro_time_gates_2 = channels_spec_norm_ch2

                # Putting together name requires some logic
                if channel_1 == channel_2 and micro_time_gates_1 == micro_time_gates_2:
                    # "Proper" autocorrelation, with the same micro time configuration
                     out_path_full += '_anom_ACF_rem_ch' + ''.join([str(element) for element in channel_1])
                    
                elif channel_1 != channel_2 and self._cross_corr_symm:
                    # Cross-correlation assuming time symmetry
                     out_path_full += '_anom_CCF_rem_symm_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
                    
                elif channel_1 == channel_2 and micro_time_gates_1 != micro_time_gates_2 and not self._cross_corr_symm:
                    # Cross-correlation of distinct micro time bins within the same channel
                     out_path_full += '_anom_CCF_rem_Microt_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
                    
                elif channel_1 == channel_2 and micro_time_gates_1 != micro_time_gates_2 and self._cross_corr_symm: 
                    # Cross-correlation of distinct micro time bins within the same channel, assuming time symmetry
                     out_path_full += '_anom_CCF_rem_Microt_symm_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
                    
                else:
                    # Cross-correlation not assuming time symmetry
                     out_path_full += '_anom_CCF_rem_ch' + ''.join([str(element) for element in channel_1]) + '_ch' + ''.join([str(element) for element in channel_2])
                
                # Update name according to applied corrections
                out_path_full += ('_br' if use_burst_removal else '') + \
                                 ('_dt' if use_drift_correction else '') + \
                                 ('_w' if use_ext_weights else '') + \
                                 ('_bg' if use_flcs_bg_corr else '')
                
                # Plot and save figure
                fig, ax = plt.subplots(nrows=1, ncols=1)
                # Cycle through colors

                for segment_index in usable_segments:
                    ax.semilogx(lag_times*1E-9, 
                                segment_ccs[:,segment_index],
                                marker = '', 
                                linestyle = '-', 
                                alpha = 0.7,
                                color = 'k' if segment_index in good_segments else 'r')
                    
                ax.set_title('Segment correlation functions') 
                fig.supxlabel('Correlation time [s]')
                fig.supylabel('G(\u03C4)')
                ax.set_xlim(lag_times.min() * 1E-9, lag_times.max() * 1E-9)
                plot_y_min_max = (np.percentile(segment_ccs[:,usable_segments], 3), np.percentile(segment_ccs[:,usable_segments], 97))
                ax.set_ylim(plot_y_min_max[0] / 1.2 if plot_y_min_max[0] > 0 else plot_y_min_max[0] * 1.2,
                            plot_y_min_max[1] * 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] / 1.2)
                
                plt.text(x = (lag_times.min()*lag_times.max()*1E-18)**(1/2), # plt.text() has weird positioning 
                         y = plot_y_min_max[1] / 1.2 if plot_y_min_max[1] > 0 else plot_y_min_max[1] * 1.2,
                         s = '''Black: Good segments \nRed: Bad segments''')
                         
                plt.savefig(out_path_full + '.png', dpi=300)
                plt.close()

                # Create and save spreadsheet
                out_table = pd.DataFrame(data ={'Lagtime[s]':lag_times*1E-9})
                
                for segment_index in range(segment_ccs.shape[1]):
                    out_table['CF_segment' + str(segment_index) + ('_good' if segment_index in good_segments else '_bad')] = segment_ccs[:, segment_index]
                    
                out_table.to_csv(out_path_full + '.csv', 
                                 index = False, 
                                 header = self._include_header)

            # Write to log, if desired
            if self._write_log and not suppress_logging:
                self.write_to_logfile(log_header = '''run_mse_filter: Segment photon stream into time windows, correlate them, and remove outliers with anomalous correlation function.''',
                                      log_message = log_comment + f'''
                                      Split data into {segment_ccs.shape[1]} segments, of which {len(usable_segments)} were worth looking at, and {good_segments.shape[0]} were kept in the end.
                                      '''+ (f'''Wrote results to {out_path_full}.csv/png''' if self._write_results else ''),
                                      calling_function = calling_function)
                
            # Whether or not we updated the macro times and weights, now we are done
            # and return the MSE matrix...Not important under normal circumstances 
            # to be honest, but it can be useful feedback for looking at the 
            # behavior of the data
            return mean_sq_error, good_segments
    
    
    #TODO: Add FLCS background subtraction
    #%% Instance methods - TCSPC, FLCS, AND BACKGROUND REMOVAL
    
    def get_micro_time_mask(self,
                            channels_spec):
        '''
        Interpret the information in the second part of a channels_spec object
        (the micro time gating) to create a boolean mask of which TCSPC bins to 
        use and which not.

        Parameters
        ----------
        channels_spec: 
            Channel configuration specifier for the which photons to correct. 
            See description in self.check_channels_spec() for details.

        Returns
        -------
        micro_time_mask : 
            np.array (1D) of bools, specifying whether or not this TCSPC bin is
            to be included in processing according to channels_spec.

        '''
        
        channels_spec_norm = self.check_channels_spec(channels_spec)
        
        micro_time_mask = np.zeros((self._n_micro_time_bins), dtype=np.bool8)

        micro_time_cutoffs = channels_spec_norm[1][0]
        micro_time_gate_indx = channels_spec_norm[1][1]

        # Apply micro time gates
        if micro_time_cutoffs == (): 
            # Empty tuple: Use all
            micro_time_mask[:] = 1
            
        elif len(micro_time_cutoffs) == 1: 

            # Single cutoff: binary
            gates = np.ceil([0, micro_time_cutoffs[0] * self._n_micro_time_bins, self._n_micro_time_bins]).astype(np.uint64)
            
            if len(micro_time_gate_indx) == 1: 
                # Use only one gate
                micro_time_mask[gates[micro_time_gate_indx[0]]:gates[micro_time_gate_indx[0]+1]] = True
                
            else: 
                # Use multiple gates
                for gate_index in micro_time_gate_indx:
                    micro_time_mask[gates[gate_index]:gates[gate_index+1]] = True
                    
        else: 

            # Muliple cutoffs
            gates = [0.]
            
            for gate in micro_time_cutoffs:
                gates.append(gate)
                
            gates.append(1)
            gates = np.ceil(gates * self._n_micro_time_bins).astype(np.uint64)
            
            if len(micro_time_gate_indx) == 1: 
                # Use only one gate
                micro_time_mask[gates[micro_time_gate_indx[0]]:gates[micro_time_gate_indx[0]+1]] = True
                
            else: 
                # Use multiple gates
                for gate_index in micro_time_gate_indx:
                    micro_time_mask[gates[gate_index]:gates[gate_index+1]] = True
        
        return micro_time_mask
    
    
    def get_tcspc_histogram(self,
                            channels_spec,
                            micro_times = [],
                            ext_indices = np.array([]),
                            use_ext_weights = False,
                            use_drift_correction = False,
                            use_burst_removal = False,
                            use_mse_filter = False,
                            suppress_logging = False,
                            calling_function = ''
                            ):
        '''
        Calculate a micro time (TCSPC) histogram, applying filters. There are 
        two ways to do this:
            1. Supplying a channels_spec structure (actually always required, 
               although depending on the details, it may not be used). This is
               the way to go if you want to get a micro time histogram from 
               the TTTR data loaded into the FCS_Fixer instance.
           2.  Supplying a micro_times array with actual micro time tags to 
               histogram. This overwrites channels_spec. This is a handle you 
               can use to get a micro time histogram of some other photons or using 
               you own custom filters or photon selection logic. This will also 
               mean that none of the built-in filters are used.

        Parameters
        ----------
        channels_spec: 
            Channel configuration specifier for the which photons to correct. 
            See description in self.check_channels_spec() for details.
            Only used if micro_times is not specified.
        micro_times : 
            OPTIONAL np.array (1D) of micro time tags (in bins, not in ns) for 
            histogramming. Overwrites channels_spec. If not supplied, the 
            photons indicated by channels_spec (and optional filters) are used 
            to create a TCSPC histogram from the loaded TTTR object.
        ext_indices :
            OPTIONAL np.array Externally specified indices of photons in the 
            self.photon_data tttr object to use.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. OPTIONAL. If False (default), ones will be 
            imputed (no specific weighting).
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        tcspc_x : 
            np.array (1D) with time axis of TCSPC histogram (in units of TCSPC
            bins, not actual time units!)
        tcspc_y : 
            np.array (1D) with binned photon counts of TCSPC histogram.

        '''
        
        channels_spec_norm = self.check_channels_spec(channels_spec)

        if micro_times == []:
            # Auto-select photons applying whatever filters have been specified previously
            use_external_micro_times = False
            _, micro_times, weights, _ = self.select_photons(channels_spec_norm, 
                                                            ext_indices = ext_indices,
                                                            use_ext_weights = use_ext_weights, 
                                                            use_drift_correction = use_drift_correction,
                                                            use_burst_removal = use_burst_removal,
                                                            use_mse_filter = use_mse_filter,
                                                            suppress_logging = suppress_logging,
                                                            calling_function = 'get_tcspc_histogram')
            
        elif not (isiterable(micro_times) and isint(micro_times[0]) and np.all(micro_times >= 0)):
            # If not auto-chosen, these three criteria are some indicator (albeit not conclusive) of whether the input makes sense
            raise ValueError('Invalid input for micro_times. Must be np.array of int >= 0 and <= number of micro time bins; or empty if you want automatically to use the loaded TTTR data.')
            
        else:
            # Implies that micro time tags that look reasonable have been externally supplied
            use_external_micro_times = True
            
            # In this case, we use dummy weights - weighting of externally supplied photons currently not supported.
            weights = np.ones_like(micro_times, dtype = np.float64)
                
        # Construct TCSPC histogram
        tcspc_x_raw = np.arange(0, self._n_micro_time_bins)
        tcspc_y_raw = np.histogram(micro_times,
                                   bins = np.append(tcspc_x_raw, self._n_micro_time_bins + 1),
                                   density = False,
                                   weights = weights)[0]
        
        
        if use_external_micro_times:
                        
            micro_time_mask = self.get_micro_time_mask(channels_spec_norm)
        
        else:
            # Implies use_external_micro_times == False, in which case select_photons() already did all the above and we merely need to discard zeros.
            # This logic technically will on occasion discard bins that by pure coincidence had zero photons, but then again, these are anyway a nuisance in TCSPC fitting.
            micro_time_mask = tcspc_y_raw > 0
            
        # Apply mask
        tcspc_x = tcspc_x_raw[micro_time_mask]
        tcspc_y = tcspc_y_raw[micro_time_mask]

        tcspc_x_ns = tcspc_x * self._micro_time_resolution

        # Write results files, if desired
        if self._write_results:
            
            # Write results to 
            out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
            self._out_file_counter += 1
            
            out_path_full += '_TCSPC_ch' + ''.join([str(element) for element in channels_spec_norm[0]]) 

            # Append out_path_full according to applied corrections
            out_path_full += ('_br' if use_burst_removal else '') + \
                             ('_dt' if use_drift_correction else '') + \
                             ('_ar' if use_mse_filter else '') + \
                             ('_w' if use_ext_weights else '')


            # Plot and save figure
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.semilogy(tcspc_x_ns, 
                        tcspc_y, 
                        '.k')
            ax.set_title('TCSPC of channel(s) [' + ','.join([str(element) for element in channels_spec_norm[0]]) + f'] at {np.round(self._micro_time_resolution*1E3)} ps resolution') 
            
            fig.supxlabel('Time [ns]')
            fig.supylabel('Counts in bin')
            ax.set_xlim(np.floor(tcspc_x_ns).min(), np.ceil(tcspc_x_ns).max())
            ax.set_ylim(1, np.max(tcspc_y) * 1.25)

            plt.savefig(out_path_full + '.png', dpi=300)
            plt.close()

            # Create spreadsheet
            out_table = pd.DataFrame(data ={'Time[ns]': tcspc_x_ns,
                                            'Counts': np.uint32(tcspc_y)})
            out_table.to_csv(out_path_full + '.csv', 
                             index = False, 
                             header = self._include_header)

        # Write to log, if desired
        if self._write_log and not suppress_logging:
            log_comment_micro_times = ('externally supplied ') if use_external_micro_times else ('from loaded TTTR retrieved ')
            log_comment_micro_times += f' {micro_times.shape[0]} photon time tags, of which {tcspc_y.sum() / micro_times.shape[0] * 100} % were ultimately used.'
                
            self.write_to_logfile(log_header = '''get_tcspc_histogram: Calculate a TCSPC histogram.''',
                                  log_message = f'''Parameters used:
                                    channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm}                                      
                                    micro_times (time tags to histogram): ''' + log_comment_micro_times + f'''
                                    ext_indices specified: {ext_indices.shape[0] > 0}
                                    use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                                    use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
                                    use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                                    use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                                    
                                    Calculated TCSPC histogram:
                                    has {tcspc_x.shape[0]} bins at {np.round(self._micro_time_resolution*1E3)} ps resolution
                                    contains {tcspc_y.sum()} photons
                                    ranges over {tcspc_x_ns.max() - tcspc_x_ns.min()} ns
                                    
                                    ''' + (f'''Wrote results to {out_path_full}.csv/png''' if self._write_results else ''),
                                    calling_function = calling_function)

        return tcspc_x, tcspc_y
        
    
    
    def find_IRF_position(self,
                          channels_spec,
                          irf_TTTR,
                          suppress_logging = False,
                          calling_function = ''
                          ):
        
        '''
        Perform a Gaussian approximate fit to the IRF in order to estimate its 
        peak position.

        Parameters
        ----------
        channels_spec: 
            Channel configuration specifier for the which photons to correct. 
            See description in self.check_channels_spec() for details.
        irf_TTTR: 
            tttrlib.TTTR object with photon data to use in IRF fitting.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        irf_peak_center : 
            Float. Fitted Gaussian peak position of the IRF in units of micro time bins.
        irf_peak_fwhm :
            Float. Fitted Gaussian peak full width at half maximum in units of micro time bins.
        irf_fit :
            TCSPC_quick_fit object with more or less complete information about fit.
            Returned only if return_full == True
        '''
        
        # Input check
        channels_spec_norm = self.check_channels_spec(channels_spec)
        i_channel = channels_spec_norm[0]
        
        # Get IRF TCSPC histogram
        irf_x, irf_y = self.get_tcspc_histogram(channels_spec,
                                                micro_times = irf_TTTR[irf_TTTR.get_selection_by_channel(i_channel)].micro_times,
                                                ext_indices = np.array([]),
                                                use_ext_weights = False,
                                                use_drift_correction = False,
                                                use_burst_removal = False,
                                                use_mse_filter = False,
                                                suppress_logging = suppress_logging,
                                                calling_function = 'find_IRF_position')
        
        # Perform fit
        irf_fit = TCSPC_quick_fit(x_data = irf_x,
                                    y_data = irf_y,
                                    model = 'gauss',
                                    initial_params = {'x_0': np.float64(irf_x[np.argmax(irf_y)]),
                                                      'y_0': np.min(irf_y),
                                                      'amp': np.max(irf_y)-np.min(irf_y),
                                                      'gauss_fwhm': 10.})
        
        irf_fit_params = irf_fit.run_fit()
        
        # Return result
        irf_peak_center = irf_fit_params['x_0']
        irf_peak_fwhm = irf_fit_params['gauss_fwhm']
        
        # Write results files, if desired
        if self._write_results:
            
            # Write results to 
            out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
            self._out_file_counter += 1
            
            out_path_full += '_IRF_fit' + ''.join([str(element) for element in channels_spec_norm[0]]) 

            # Plot and save figure
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.semilogy(irf_fit.x, 
                        irf_fit.y, 
                        '.k', 
                        label = 'IRF data')
            ax.semilogy(irf_fit.x, 
                        irf_fit.prediction,
                        'tab:gray',
                        label = 'Gaussian fit')
            
            ax.set_title('IRF channel(s) [' + ','.join([str(element) for element in channels_spec_norm[0]]) + '] with Gaussian approximation')
            fig.supxlabel('Time [TCSPC bins]')
            fig.supylabel('Counts in bin')
            ax.set_xlim(irf_fit.x.min(), irf_fit.x.max())
            ax.set_ylim(1, np.max(irf_fit.y) * 1.25)
            ax.legend()

            plt.savefig(out_path_full + '.png', dpi=300)
            plt.close()

            # Create spreadsheet
            out_table = pd.DataFrame(data ={'Time[ns]': irf_fit.x * self._micro_time_resolution,
                                            'Counts': np.uint32(irf_fit.y),
                                            'Fit': irf_fit.prediction})
            
            out_table.to_csv(out_path_full + '.csv', 
                             index = False, 
                             header = self._include_header)
        
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''find_IRF_position: Fit a Gaussian approximation to an IRF to determine its center and width.''',
                                  log_message = f'''Parameters used:
                                      channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm}
                                      irf_TTTR: tttrlib.TTTR object containg measurement IRF data 
                                          -> {irf_TTTR.header.macro_time_resolution * np.max(irf_TTTR.macro_times) * 1E-9} s acquisition time
                                          -> {irf_y.sum()} photons used out of {irf_TTTR.macro_times.shape[0]} in total.
                                          -> {irf_TTTR.header.micro_time_resolution} ns TCSPC resolution and {irf_TTTR.get_number_of_micro_time_channels()} micro time channels
                                      
                                      Estimated the following IRF parameters (reduced chi-square: {irf_fit_params['red_chi_sq']}):
                                      x_0 (IRF peak center) [micro time channels]: {irf_peak_center}
                                      gauss_fwhm (IRF peak full width at half-maximum) [TCSPC time channels]: {irf_peak_fwhm}
                                      amp (IRF peak smplitude) [counts per micro time channel]: {irf_fit_params['amp']}
                                      y_0 (laser-independent background) [counts per micro time channel]: {irf_fit_params['y_0']}
                                      
                                      ''' + (f'''Wrote results to {out_path_full}.csv/png''' if self._write_results else ''),
                                      calling_function = calling_function)        
                                      
        return irf_peak_center, irf_peak_fwhm, irf_fit
        
    
    
    def get_background_tail_fit(self,
                                   channels_spec,
                                   irf_peak_center,
                                   fit_start,
                                   ext_indices = np.array([]),
                                   use_ext_weights = False,
                                   use_drift_correction = False,
                                   use_burst_removal = False,
                                   use_mse_filter = False,
                                   suppress_logging = False,
                                   calling_function = ''
                                   ):
        '''
        Perform an exponential tail fil to the micro_time data of the TTTR 
        object to determine the background level in that channel. Note that 
        you can use all sorts of photon selection and weighting for this to 
        determine the background contaminating a specific signal controbution.

        Parameters
        ----------
        channels_spec: 
            Channel configuration specifier for the which photons to correct. 
            See description in self.check_channels_spec() for details.
        irf_peak_center : Float
            Peak position of the IRF (as TCSPC bin index), used as reference 
            for estimating the amplitude of the decay. You may want to use 
            self.find_IRF_position() to find this.
        fit_start : int
            index of the TCSPC bin that is the left edge of the fit range. 
            Given that this is intended for tail fitting, it should definitely 
            be > irf_peak_center. Depends on the data, but a good start may be 
            irf_peak_center + 5 * irf_peak_fwhm as determined from 
            self.find_IRF_position().
        ext_indices :
            OPTIONAL np.array Externally specified indices of photons in the 
            self.photon_data tttr object to use.
        use_ext_weights :
            OPTIONAL bool. Whether to use the external weights stored in 
            self._weights_ext containing photon weights from FLCS 
            filter etc. If False (default), ones will be 
            imputed (no specific weighting).
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights. Keep in mind that you would 
            typically use this function as part of the calculation to DETERMINE
            the undrift weights, so think twice about the use here.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        flat_background : Float
            Background counts per TCSPC bin.
        tail_fit : TCSPC_quick_fit object 
            Object containing a lot of information relating to the fit.
            Returned only if return_full == True

        '''
        
        # Input check
        if not isint(fit_start) or not (fit_start > irf_peak_center):
            raise ValueError('Invalid input for fit_start. Must be int > irf_peak_center.')
            
        channels_spec_norm = self.check_channels_spec(channels_spec)
        
        # Get TCSPC for fitting
        tcspc_x, tcspc_y = self.get_tcspc_histogram(channels_spec,
                                                    micro_times = [],
                                                    ext_indices = ext_indices,
                                                    use_ext_weights = use_ext_weights, 
                                                    use_drift_correction = use_drift_correction,
                                                    use_burst_removal = use_burst_removal,
                                                    use_mse_filter = use_mse_filter,
                                                    suppress_logging = suppress_logging,
                                                    calling_function = 'get_background_tail_fit')

        # Crop a little further to ensure that we perform a tail fit only, rather than fitting the peak
        tail_fit_use = tcspc_x >= fit_start
        tcspc_x_crop = tcspc_x[tail_fit_use]
        tcspc_y_crop = tcspc_y[tail_fit_use]

        # Fit
        tail_fit = TCSPC_quick_fit(x_data = tcspc_x_crop,
                                    y_data = tcspc_y_crop,
                                    model = 'exponential',
                                    initial_params = {'x_0': irf_peak_center,
                                                      'y_0': np.min(tcspc_y_crop),
                                                      'amp': np.max(tcspc_y)-np.min(tcspc_y), # This refers to the peak, so we estimate the initial parameter from the uncropped data
                                                      'exp_tau': 100.})
        tail_fit_params = tail_fit.run_fit()
        
        # Return
        flat_background = tail_fit_params['y_0']

        # Write results files, if desired
        if self._write_results:
            
            tcspc_x_crop_ns = tail_fit.x * self._micro_time_resolution
            
            out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
            self._out_file_counter += 1
            
            out_path_full += '_Tail_fit' + ''.join([str(element) for element in channels_spec_norm[0]]) 

            # Append out_path_full according to applied corrections
            out_path_full += ('_br' if use_burst_removal else '') + \
                             ('_dt' if use_drift_correction else '') + \
                             ('_ar' if use_mse_filter else '') + \
                             ('_w' if use_ext_weights else '')

            # Plot and save figure
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.semilogy(tcspc_x_crop_ns, 
                        tail_fit.y, 
                        '.k', 
                        label = 'Data')
            ax.semilogy(tcspc_x_crop_ns, 
                        tail_fit.prediction, 
                        'tab:gray', 
                        label = 'Tail fit')
            ax.semilogy([tcspc_x_crop_ns.min(), tcspc_x_crop_ns.max()], 
                        [flat_background, flat_background], 
                        color = 'tab:gray', 
                        linestyle = ':', 
                        label = 'Background level')

            ax.set_title('TCSPC channel(s) [' + ','.join([str(element) for element in channels_spec_norm[0]]) + '] with exponential tail fit')
            fig.supxlabel('Time [ns]')
            fig.supylabel('Counts in bin')
            ax.set_xlim(np.floor(tcspc_x_crop_ns).min(), np.ceil(tcspc_x_crop_ns).max())
            ax.set_ylim(1, np.max(tail_fit.y) * 1.25)
            ax.legend()

            plt.savefig(out_path_full + '.png', dpi=300)
            plt.close()

            # Create spreadsheet
            out_table = pd.DataFrame(data ={'Time[ns]': tcspc_x_crop_ns,
                                            'Counts': np.uint32(tail_fit.y),
                                            'Fit': tail_fit.prediction})
            
            out_table.to_csv(out_path_full + '.csv', 
                             index = False, 
                             header = self._include_header)

        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''get_background_tail_fit: Fit an exponential decay to the tail of a TCSPC histogram to determine the flat background level.''',
                                  log_message = f'''Parameters used:
                                      channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm}
                                      irf_peak_center (as reference point for the amplitude) [micro time channels]: {irf_peak_center}
                                      fit_start (left edge of time window to include in fit) [micro time channels]: {fit_start}
                                      ext_indices specified: {ext_indices.shape[0] > 0}
                                      use_ext_weights (use of weights for FLCS etc.): {use_ext_weights}
                                      use_drift_correction (use of weights for bleaching/drift correction etc.): {use_drift_correction}
                                      use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                                      use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}

                                      Included {tcspc_y.sum()} photons in fit.
                                      Fitted the following parameters (reduced chi-square: {tail_fit_params['red_chi_sq']}):
                                      y_0 (laser-independent background) [counts per micro time channel]: {flat_background}
                                      exp_tau (1/e decay time of the exponential) [TCSPC time channels]: {tail_fit_params['exp_tau']}
                                      amp (exponential decay amplitude) [counts per micro time channel]: {tail_fit_params['amp']}
                                      
                                      ''' + (f'''Wrote results to {out_path_full}.csv/png''' if self._write_results else ''),
                                      calling_function = calling_function)        
                                      
        return flat_background, tail_fit
    
    
    def get_flcs_background_filter(self,
                                   tcspc_x,
                                   tcspc_y,
                                   flat_background,
                                   channels_spec,
                                   handle_outside = 'zero',
                                   update_weights = True,
                                   ext_indices = np.array([]),
                                   suppress_logging = False,
                                   calling_function = ''
                                   ):
        '''
        Wrapper for self.get_flcs_filters() for correcting only a flat background
        contribution. Includes automatic writing of results to .csv and .png files.

        Parameters
        ----------
        tcspc_x : 
            np.array (1D) with time axis of TCSPC histogram (in units of TCSPC
            bins, not actual time units!)
        tcspc_y : 
            np.array (1D) with binned photon counts of TCSPC histogram.
        flat_background : 
            Float specifying flat background level to correct (in counts per
            micro time bin).
        channels_spec: 
            Channel configuration specifier for which photons to correct. 
            See description in self.check_channels_spec() for details.
        handle_outside : 
            OPTIONAL string with default 'zero', alternatively 'ignore'. If 
            tcspc_x does not cover the full micro time range specified by 
            channels_spec for whatever reason, what to do to the weights of 
            photons outside tcspc_x? If 'zero', these photons are effectively 
            discarded. If 'ignore', whatever weights these photons had before
            (by default 1) are kept as-is. Basically, you can use 'ignore' to
            run multiple background correction rounds specifying different 
            background levels for different micro time windows. Not entirely 
            sure if anybody will ever need that, but well, it's possible. 
            Honestly, this is something that exists because the workflow of this
            method ended up being slightly different from how I had initially envisioned it...
        update_weights : 
            OPTIONAL bool with default True. Whether to write the filter weights 
            from FLCS background correction to property self._weights_flcs_bg_corr.
        ext_indices : TYPE, optional
            DESCRIPTION. The default is np.array([]).
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.


        Returns
        -------
        patterns_norm_full : 
            np.array (2D). Normalized micro time patterns for signal 
            (patterns_norm_full[:,0]) and background (patterns_norm_full[:,1]).
            Irrespective of the lenth of tcspc_x and tcspc_y, the length along 
            axis 0 covers the full micro time range of the loaded TTTR data. 
            TCSPC bins that were not used in the filter calculation contain zeros.
        flcs_weights_full : 
            np.array (2D). flcs_weights_full[:,0] contains filter weights for the 
            actual signal, flcs_weights_full[:,1] weights for the background.
            Irrespective of the lenth of tcspc_x and tcspc_y, the length along 
            axis 0 covers the full micro time range of the loaded TTTR data. 
            TCSPC bins that were not used in the filter calculation contain zeros.
        '''
        
        # Input check
        if not (isiterable(tcspc_x) and isint(tcspc_x[0])):
            raise ValueError('Invalid input for tcspc_x: Must be np.ndarray of int, specifying indices (not time tags!) of TCSPC bins.')
            
        else:
            # Make sure we get the type consistent
            tcspc_x = np.array(tcspc_x, 
                               dtype = np.uint64)
            
        if not (isiterable(tcspc_y) and len(tcspc_y) == len(tcspc_x)):
            raise ValueError('Invalid input for tcspc_y: Must be np.ndarray of same length as tcspc_x.')
            
        else:
            # Make sure we get the type consistent
            tcspc_y = np.array(tcspc_y, 
                               dtype = np.float64)

        channels_spec_norm = self.check_channels_spec(channels_spec)
        
        if not handle_outside in ['zero', 'ignore']:
            raise ValueError('Invalid input for handle_outside. Must be "zero" (discard photons outside tcspc_x) or "ignore" (leave photons outside tcspc_x unchanged).')
        
        # Get (non-normalized) patterns
        pattern_signal = tcspc_y - flat_background
        pattern_background = np.ones_like(tcspc_y)
        
        # We exclude zeros from filter calculation, and add a second dimension to the patterns which we'll need in a sec
        nonzero_mask = np.logical_and(tcspc_y != 0, pattern_signal != 0)
        n_nonzeros = nonzero_mask.sum()
        
        tcspc_y_crop = tcspc_y[nonzero_mask]
        pattern_signal_crop = pattern_signal[nonzero_mask].reshape((n_nonzeros, 1))
        pattern_background_crop = pattern_background[nonzero_mask].reshape((n_nonzeros, 1))
        
        # Get filters
        flcs_weights_crop, patterns_norm_crop = self.get_flcs_filters(tcspc_y = tcspc_y_crop,
                                                                    patterns = np.concatenate((pattern_signal_crop, pattern_background_crop), 
                                                                                              axis = 1))
                
        # Expand into full TCSPC range
        tcspc_x_full = np.arange(self._n_micro_time_bins)
                
        patterns_norm_full = np.zeros((self._n_micro_time_bins, 2),
                                      dtype = np.float64)
        patterns_norm_full[tcspc_x[nonzero_mask], 0] = patterns_norm_crop[:, 0]
        patterns_norm_full[tcspc_x[nonzero_mask], 1] = patterns_norm_crop[:, 1]
        
        flcs_weights_full = np.zeros((self._n_micro_time_bins, 2), 
                                     dtype = np.float64)
        flcs_weights_full[tcspc_x[nonzero_mask], 0] = flcs_weights_crop[:, 0]
        flcs_weights_full[tcspc_x[nonzero_mask], 1] = flcs_weights_crop[:, 1]

        
        _, micro_times_select, weights_select, indices_select = self.select_photons(channels_spec_norm, 
                                                                                    ext_indices = ext_indices,
                                                                                    use_ext_weights = False, 
                                                                                    use_drift_correction = False,
                                                                                    use_flcs_bg_corr = True, # We retrieve the already-written FLCS bg correction weights 
                                                                                    use_burst_removal = False,
                                                                                    use_mse_filter = False,
                                                                                    suppress_logging = suppress_logging,
                                                                                    calling_function = 'get_flcs_background_filter')
        
        if update_weights:
            # Annotate on a photon-by-photon basis
            flcs_photon_weights = np.zeros_like(weights_select)
            
            for i_bin, micro_time in enumerate(tcspc_x[nonzero_mask]):
                flcs_photon_weights[micro_times_select == micro_time] = flcs_weights_crop[i_bin, 0]
    
            # Now we need a slightly complicated logical expression. This checks if 
            # tcspc_x covers everything within the micro time range specified in channels_spec.
            micro_time_mask = self.get_micro_time_mask(channels_spec_norm)
            tcspc_x_mask = np.zeros((self._n_micro_time_bins), dtype = np.bool8)
            tcspc_x_mask[tcspc_x] = True
            
            if np.any(np.logical_xor(tcspc_x_mask, micro_time_mask)):
                # If True, tcspc_x did not include all bins, so we need to decide 
                # what to do with the rest. 
    
                if handle_outside == 'ignore':
                    # We leave photons outside tcspc_x untouched, i.e., we copy their previous weights
                    log_comment_handle_outside = "'ignore' (leave photons outside tcspc_x unchanged)"
                    # Find which bins to retrieve from previous data
                    micro_times_used = np.unique(micro_times_select)
                    not_updated_micro_times = np.logical_not(np.isin(micro_times_used, tcspc_x))
                    
                    # Update
                    for micro_time in not_updated_micro_times:
                        mask = micro_times_select == micro_time
                        flcs_photon_weights[mask] = weights_select[mask]
    
                else:
                    # implies handle_outside == 'zero':
                    # We effectively dicard all photons outside tcspc_x by leaving 
                    # their weights at 0, which effectively means "do nothing further, except write to log".
                    log_comment_handle_outside = "'zero' (discard photons outside tcspc_x)"
    
    
            else:
                # implies that tcspc_x did cover the whole range specified by 
                # channels_spec. Then it's easier - actually nothing to do, except logging
                log_comment_handle_outside = "irrelavant, full histogram was covered"


            # Write to property
            self._weights_flcs_bg_corr[indices_select] = flcs_photon_weights
            
        else:
            # update_weights == False, in which case much of the calculation is skipped.
            log_comment_handle_outside = "irrelavant, photon-wise weights were not updated"
            
        # Write results files, if desired
        if self._write_results:
                                       
            out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
            self._out_file_counter += 1
            
            out_path_full += '_FLCS_bg_corr_filters' 

            # Figure
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex = True)
                            
            # Top panel: Normalized patterns
            ax[0].semilogy(tcspc_x_full,
                           patterns_norm_full[:,0], 
                           marker = '.',
                           linestyle = 'none',
                           color = 'k',
                           label = 'Signal')
            
            ax[0].semilogy(tcspc_x_full,
                           patterns_norm_full[:,1], 
                           marker = '.',
                           linestyle = 'none',
                           color = 'tab:gray',
                           label = 'Background')
            
            ax[0].set_title('Normalized micro time patterns')
            for_y_scaling = np.percentile(patterns_norm_full[patterns_norm_full > 0], 1.5)
            ax[0].set_ylim(for_y_scaling * 0.5, np.max(patterns_norm_full) * 1.25)
            ax[0].legend()
            
            # Bottom panel: Filter functions
            ax[1].plot(tcspc_x_full,
                        flcs_weights_full[:,0], 
                        marker = '',
                        linestyle = '-',
                        color = 'k',
                        label = 'Signal')
            
            ax[1].plot(tcspc_x_full,
                        flcs_weights_full[:,1], 
                        marker = '',
                        linestyle = '-',
                        color = 'tab:gray',
                        label = 'Background')
                
            ax[1].set_title('FLCS filter functions')
            ax[1].set_ylim(np.min(flcs_weights_full) * 1.1, np.max(flcs_weights_full) * 1.1) # both *1.1 as there will be negative numbers
            ax[0].set_xlim(0, self._n_micro_time_bins) # Applies to both panels
            fig.supxlabel('TCSPC bins')

            plt.savefig(out_path_full + '.png', dpi=300)
            plt.close()

            # Create spreadsheet
            tcspc_y_full = np.zeros((self._n_micro_time_bins), 
                                         dtype = np.float64)
            tcspc_y_full[tcspc_x[nonzero_mask]] = tcspc_y
            
            out_table = pd.DataFrame(data ={'Index': tcspc_x_full,
                                            'Sum_TCSPC': np.uint32(tcspc_y_full),
                                            'Signal_pattern': patterns_norm_full[:,0],
                                            'Background_pattern': patterns_norm_full[:,1],
                                            'Signal_filter': flcs_weights_full[:,0],
                                            'Background_filter': flcs_weights_full[:,1],
                                            })
            
            out_table.to_csv(out_path_full + '.csv', 
                             index = False, 
                             header = self._include_header)

        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''get_flcs_background_filter: Use a previously determined (flat) background level to determine FLCS/fFCS filters for background suppression.''',
                                  log_message = f'''Parameters used:
                                      tcspc_x (TCSPC bin indices) and tcspc_y (full TCSPC histogram corresponding to those bins): {tcspc_y.shape[0]} bins, among those {np.sum(tcspc_y == 0)} empty bins, {tcspc_y.sum()} photons
                                      flat_background (flat background level) [counts per TCSPC bin]: {flat_background}
                                      channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec_norm}
                                      handle_outside (what to do with photons outside in bins tcspc_x: ''' + log_comment_handle_outside + '''
                                      update_weights (whether to write updated weights to property in addition to returnig): {update_weights}
                                      ext_indices specified: {ext_indices.shape[0] > 0}
                                      
                                      ''' + (f'''Wrote results to {out_path_full}.csv/png''' if self._write_results else ''),
                                      calling_function = calling_function)        
                          
        return patterns_norm_full, flcs_weights_full

    #%% Instance methods - Photon counting histogram

    def get_PCH(self,
                channels_spec,
                bin_time,
                normalize = False,
                ext_indices = np.array([]),
                use_burst_removal = False,
                use_mse_filter = False,
                more_channels_specs = None,
                suppress_logging = False,
                calling_function = ''
                ):
        '''
        Calculates a photon counting histogram for a channel of interest,
        including all the selection options in channels_spec, the option to externally
        specify which photons to use and which not, burst removal and the MSE-based
        removal of anomalous data segments. Note that PCH is not used with photon 
        weighting-based filters, as these would lead to a broadening of the PCH 
        and thus distortion of the parameters!
                
        Parameters
        ----------
        channels_spec: 
            Channel configuration specifier for the correlation operation. 
            See description in self.check_channels_spec() for details. 
        bin_time:
            Float specifying the desired bin width in seconds.
        normalize:
            OPTIONAL Bool with default False, specifying whether or not to 
            normalize the PCH.
        ext_indices :
            OPTIONAL np.array Externally specified indices of photons in the self.photon_data 
            tttr object to use in correlation. Can be used in slicing 
            data into a series of correlation fucntions, or to 
            randomize stuff (bootstrap). Or the user can exploit it as a handle
            to implement their own custom photon selection logic, obviously.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        more_channels_specs :
            OPTIONAL list of channels_spec, default None. Offers the option to 
            enter multiple channels_spec objects to create a PCH of a sum channel.
            Please note that using this option currently is unsafe when use_burst_removal
            or use_mse_filter is true, as the time traces may end up with different
            excesed segments.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the Class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        pch :
            np.array containing counts (or probability mass depending on arg 
            "normalize") of PCH. Corresponding a-axis is not explicitly
            returned, it would be np.arange(0, pch.shape[0])
        Mandel_Q :
            Mandel's Q parameter for the data at the given binning. Check 
            literature for details, basically a quick indicator if the data is
            Poisson-like (Q ~ 1, typically not too useful for PCH), 
            super-Poissonian (Q > 1, desirable for PCH), or sub-Poissonian 
            (Q < 1, a sign of strong detector dead time artifacts)
        '''
        
        # Unpack required channels information
        channels_spec = self.check_channels_spec(channels_spec)
        
        # Get time trace - we do not really care about the time inforamtion in PCH,
        # but the tt is an intermediate 
        time_trace, _ = self.get_time_trace(channels_spec, 
                                             time_trace_sampling = bin_time,
                                             use_ext_weights = False,
                                             ext_indices = ext_indices,
                                             use_drift_correction = False,
                                             use_flcs_bg_corr = False,
                                             use_burst_removal = use_burst_removal,
                                             use_mse_filter = use_mse_filter,
                                             suppress_writing = True, # here we suppress the time trace writing
                                             suppress_logging = suppress_logging,
                                             calling_function = 'get_PCH')
        
        if type(more_channels_specs) == list and len(more_channels_specs) > 0:
            for channels_spec_add in more_channels_specs:
                
                # Check if entry is valid
                channels_spec_add_norm = self.check_channels_spec(channels_spec_add)

                time_trace_new, _ = self.get_time_trace(channels_spec_add_norm, 
                                                         time_trace_sampling = bin_time,
                                                         use_ext_weights = False,
                                                         ext_indices = ext_indices,
                                                         use_drift_correction = False,
                                                         use_flcs_bg_corr = False,
                                                         use_burst_removal = use_burst_removal,
                                                         use_mse_filter = use_mse_filter,
                                                         suppress_writing = True, # here we suppress the time trace writing
                                                         suppress_logging = suppress_logging,
                                                         calling_function = 'get_PCH')
                
                time_trace += time_trace_new

        photon_count_maximum = np.max(time_trace)
        # Get actual PCH
        pch, _ = np.histogram(time_trace,
                              bins = np.arange(0, photon_count_maximum + 1),
                              density = normalize)
        
        photon_count_variance = np.var(time_trace)
        photon_count_mean = np.mean(time_trace)
        Mandel_Q = photon_count_variance / photon_count_mean - 1
        
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''get_PCH: Calculate a photon counting histogram.''',
                                  log_message = f'''Parameters used:
                                    channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec}
                                    bin_time [s]: {bin_time}
                                    normalize: {normalize}
                                    ext_indices specified: {ext_indices.shape[0] > 0}
                                    use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                                    use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                                    
                                    Found maximum of {photon_count_maximum} photons in bin
                                    Histogram is characterized by Mandel's Q of {Mandel_Q}
                                    ''',
                                    calling_function = calling_function)
        
        return pch, Mandel_Q
    
    
    def get_PCMH(self,
                channels_spec,
                spacing = 2.,
                normalize = False,
                ext_indices = np.array([]),
                use_burst_removal = False,
                use_mse_filter = False,
                more_channels_specs = None,
                suppress_logging = False,
                calling_function = ''
                ):
        '''
        Calculate a series of photon counting histograms with equidistant 
        logarithmic spacing of bin widths, starting at 1 us and ending at 
        at the highest bin width in the logarithmic spacing that yielded more 
        than 1000 independent bins.
        See also "Photon Counting Multiple Histograms" as described in:
        Perroud, Huang, Zare ChemPhysChem 2005 DOI: 10.1002/cphc.200400547

        Parameters
        ----------
        channels_spec: 
            Channel configuration specifier for the correlation operation. 
            See description in self.check_channels_spec() for details.
        spacing : 
            OPTIONAL float with default np.sqrt(2.). The logarithmic spacing
            at which the PCH is recalculated repeatedly.
        normalize : 
            OPTIONAL bool with default False. Whether or not to return the PCHs
            in normalized form.
        ext_indices :
            OPTIONAL np.array Externally specified indices of photons in the self.photon_data 
            tttr object to use in correlation. Can be used in slicing 
            data into a series of correlation fucntions, or to 
            randomize stuff (bootstrap). Or the user can exploit it as a handle
            to implement their own custom photon selection logic, obviously.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        more_channels_specs :
            OPTIONAL list of channels_spec, default None. Offers the option to 
            enter multiple channels_spec objects to create a PCH of a sum channel.
            Please note that using this option currently is unsafe when use_burst_removal
            or use_mse_filter is true, as the time traces may end up with different
            excesed segments.
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the Class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.

        Returns
        -------
        bin_times :
            np.array (1D) with all bin times used in seconds.
        pcmh : 
            np.array (2D) with PCHs. Each PCH is one column along axis 0, with 
            the index along the axis giving the photon number represented by 
            the bin. Axis two is the sequence of PCHs over bin widths.
        Mandel_Q_series :
            Mandel's Q parameter for the data at each bin time. Check 
            literature for details, basically a quick indicator if the data is
            Poisson-like (Q ~ 1, typically not too useful for PCH), 
            super-Poissonian (Q > 1, desirable for PCH), or sub-Poissonian 
            (Q < 1, a sign of strong detector dead time artifacts)

        '''
        channels_spec = self.check_channels_spec(channels_spec)
        
        if spacing <= 1.:
            raise ValueError('spacing must be greater 1!')
        
        # We use all bin widths starting at 1 us going up in factors of "spacing"
        # until the last one where we get 1000 bins out of the data (ignoring 
        # possible loss due to MSE filter and whatnot)
        max_bin_time = self.acquisition_time / 1E4
        
        bin_times = [1E-6]
        next_bin_time = 1E-6 * spacing
        while next_bin_time < max_bin_time:
            bin_times.append(next_bin_time)
            next_bin_time *= spacing
        bin_times = np.array(bin_times)
        
        # We get the last PCH with the widest bins first, as it defined the 
        # pd.DF size we'll need (widest bins -> reaches the highest photon counts)
        pch, Mandel_Q = self.get_PCH(channels_spec,
                                   bin_time = bin_times[-1],
                                   normalize = normalize,
                                   ext_indices = ext_indices,
                                   use_burst_removal = use_burst_removal,
                                   use_mse_filter = use_mse_filter,
                                   more_channels_specs = more_channels_specs,
                                   suppress_logging = suppress_logging,
                                   calling_function = 'get_PCMH'
                                   )
        
        pcmh = np.zeros((pch.shape[0], len(bin_times)))
        pcmh[:,-1] = pch
        
        Mandel_Q_series = np.zeros(len(bin_times))
        Mandel_Q_series[-1] = Mandel_Q
        
        for i_bin_time, bin_time in enumerate(bin_times[:-1]):
            pch, Mandel_Q = self.get_PCH(channels_spec,
                                        bin_time = bin_time,
                                        normalize = normalize,
                                        ext_indices = ext_indices,
                                        use_burst_removal = use_burst_removal,
                                        use_mse_filter = use_mse_filter,
                                        more_channels_specs = more_channels_specs,
                                        suppress_logging = suppress_logging,
                                        calling_function = 'get_PCMH'
                                        )
                                        
            pcmh[0:pch.shape[0], i_bin_time] = pch
            Mandel_Q_series[i_bin_time] = Mandel_Q

        
        if self._write_results:
            # Write results to csv and png output
            out_path_full = os.path.join(self._out_path, ('0' + str(self._out_file_counter)) if self._out_file_counter < 10 else str(self._out_file_counter))
            self._out_file_counter += 1
            
            # Construct output names
            out_path_full_PCMH = out_path_full + '_PCMH' + ''.join(['_ch' +str(element) for element in channels_spec[0]])
            out_path_full_Mandel_Q = out_path_full + '_Mandel_Q' + ''.join(['_ch' + str(element) for element in channels_spec[0]])
            
            # More channels?
            if more_channels_specs != None:
                for channels_spec_add in more_channels_specs:
                    out_path_full_PCMH += ''.join(['_ch' +str(element) for element in channels_spec_add[0]])
                    out_path_full_Mandel_Q += ''.join(['_ch' + str(element) for element in channels_spec_add[0]])

            # Update name according to applied corrections
            out_path_full_PCMH += ('_br' if use_burst_removal else '') + \
                                 ('_ar' if use_mse_filter else '')            
            out_path_full_Mandel_Q += ('_br' if use_burst_removal else '') + \
                                 ('_ar' if use_mse_filter else '')            
             
            # Create and write spreadsheets
            out_table = pd.DataFrame(data = {str(bin_time): pcmh[:,i_bin_time] for i_bin_time, bin_time in enumerate(bin_times)})
            out_table.to_csv(out_path_full_PCMH + '.csv', 
                              index = False, 
                              header = True)
        
            out_table = pd.DataFrame(data = {'Bin Times [s]': bin_times,
                                             'Mandel Q': Mandel_Q_series})
            out_table.to_csv(out_path_full_Mandel_Q + '.csv', 
                              index = False, 
                              header = True)
            
            # Create and write figure
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex = False)
            pch_x = np.arange(0, pcmh.shape[0])
            
            # Left panel: PCMH
            # Cycle through colors
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = cycle(prop_cycle.by_key()['color'])

            for i_pch in range(pcmh.shape[1]):
                iter_color = next(colors)
                # Plot is nicer/easier to compare with normalized PCMH
                pch_norm = pcmh[:,i_pch] / np.sum(pcmh[:,i_pch])
                ax[0].semilogy(pch_x, 
                               pch_norm,
                               marker = '', 
                               linestyle = '-', 
                               alpha = 0.7,
                               color = iter_color)
            ax[0].set_title('Norm. PCH over bin time')
            ax[0].set_ylim(1E-6, 1)
            
            # Right panel: Mandel's Q
            ax[1].plot(bin_times,
                        Mandel_Q_series, 
                        marker = 'o',
                        linestyle = '-',
                        color = 'k')
            ax[1].set_title("Mandel's Q")
            

            plt.savefig(out_path_full_PCMH + '.png', dpi=300)
            plt.close()
            
        # Write to log, if desired
        if self._write_log and not suppress_logging:
            self.write_to_logfile(log_header = '''get_PCMH: Series of PCHs with different bin times.''',
                                  log_message = f'''Parameters used:
                                    channels_spec ((one_or_multiple_channels), ((PIE_gate_edges), (which_gates_to_use))): {channels_spec}
                                    spacing : {spacing}
                                      -> Bin times: [''' + ''.join([str(element) + ', ' for element in bin_times]) + f'''] s
                                    normalize: {normalize}
                                    ext_indices specified: {ext_indices.shape[0] > 0}
                                    use_burst_removal (removal of photons assigned as burst photons): {use_burst_removal}
                                    use_mse_filter (removal of photons in segments with anomalous correlation function): {use_mse_filter}
                                    ''' + (f'''Results written to {out_path_full_PCMH}.csv/png''' if self._write_results else ''),
                                    calling_function = calling_function)

        return bin_times, pcmh, Mandel_Q_series
        

    
    #%% large wrapper for everything
    def run_standard_pipeline(self,
                              channels_spec_1,
                              channels_spec_2,
                              use_burst_removal,
                              use_drift_correction,
                              use_mse_filter,
                              use_flcs_bg_corr,
                              default_uncertainty_method = 'Wohland',
                              write_intermediate_ccs = False,
                              write_pcmh = True,
                              calling_function = '',
                              suppress_logging = False):
        '''
        Single-call shorthand for "do just about everything this class can do".
        
        The pipeline is:
            1. Correlate without any filters
            2. Remove bursts and correlate again
            3. Also run undrift routine, and correlate again
            4. Run the MSE filter for anomalous correlation functions, and correlate again
            5. Run FLCS background correction, and correlate again
            
        You can use the Booleans to skip any (or all) of the 4 filters, but you can't 
        change the order in which they are applied. If you want to use the 
        filters in a different order, chain the lower-level functions into the 
        correct order in a custom script.
        
        Note that this function returns nothing to your Python namespace, it 
        only writes output files. It is entirely useless if the write_results
        parameter of the class instance is False!
        
        Parameters
        ----------
        channels_spec_1, channels_spec_2 :
            Channel configuration specifier for which photons to use. 
            See description in self.check_channels_spec() for details.
        use_drift_correction :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from undrifting in photon weights. Keep in mind that you would 
            typically use this function as part of the calculation to DETERMINE
            the undrift weights, so think twice about the use here.
        use_burst_removal :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._macro_time_correction_burst_removal and self._weights_burst_removal to mask out 
            photons labelled as burst photons
        use_flcs_bg_corr :
            OPTIONAL bool with default False. Whether to consider photon weights 
            from FLCS background correction in photon weights.
        use_mse_filter :
            OPTIONAL bool. Specifies whether or not to use the attributes 
            self._weights_anomalous_segments and self._macro_times_correction_mse_filter
            to mask out photons labelled as being in an anomalous time segment.
        default_uncertainty_method : 
            OPTIONAL string with default 'Wohland'. Alternative is 'Bootstrap'.
            Choice of uncertainty calculation method to be used by self.get_correlation_uncertainty()
            If 'Wohland' is chosen, self.get_Wohland_SD() is used as the default
            method of standard deviation calculation, and self.get_bootstrap_SD()
            as the backup method. If 'Bootstrap' is chosen, the software will 
            directly go to self.get_bootstrap_SD().
        write_intermediate_ccs :
            OPTIONAL bool with default False. If True, an FCS output is calculated
            and written after every filtering step. If False, FCS output is 
            caclulated only at the end (or upon crashing of the pipeline, 
            using whatever filters had been successfully prepared).
        write_pcmh :
            OPTIONAL bool with default True. Whether to calculate and export
            photon counting (multiple) histograms output in addition to FCS output. 
        suppress_logging :
            OPTIONAL bool. If the function called with suppress_logging == True, the call will 
            not be registered in the log file even if the class instance has been 
            set to create a log file. The default is False.
        calling_function : 
            string, optional This is a handle specifically meant for logging 
            which function had called this function. To make the code stack 
            more understandable in the log.
        
        Returns
        -------
        None.
        
        '''
        
        # Input check 
        if not self._write_results: 
            raise Warning('run_standard_pipeline() does not do anything useful with write_results == False!')
            
        channels_spec_norm_ch1 = self.check_channels_spec(channels_spec_1)
        channels_spec_norm_ch2 = self.check_channels_spec(channels_spec_2)
        
        # Is this auto- or cross-correlation? In the former case, we can skip some calculations.
        is_cross_corr = channels_spec_1 != channels_spec_2
        
        if write_intermediate_ccs or not (use_burst_removal or use_drift_correction or use_flcs_bg_corr or use_mse_filter):
        # First correlation: No filters
            _ = self.get_correlation_uncertainty(channels_spec_norm_ch1,
                                                 channels_spec_norm_ch2, 
                                                 use_drift_correction = False,
                                                 use_flcs_bg_corr = False,
                                                 use_burst_removal = False,
                                                 use_mse_filter = False,
                                                 default_uncertainty_method = default_uncertainty_method,
                                                 calling_function = 'run_standard_pipeline',
                                                 suppress_logging = suppress_logging)
        
        # First filter: Burst removal
        if use_burst_removal:
            
            try:
                # Auto-tune time trace bin width
                time_trace_sampling = self.get_trace_time_scale(channels_spec_norm_ch1,
                                                                calling_function = 'run_standard_pipeline',
                                                                suppress_logging = suppress_logging)
                
                if is_cross_corr:
                    # If we have two-channel data, let's use a geometric mean of 
                    # time_trace_sampling suggestions for two the two channels as compromise
                    time_trace_sampling = np.sqrt(time_trace_sampling * self.get_trace_time_scale(channels_spec_norm_ch2,
                                                                                                  calling_function = 'run_standard_pipeline',
                                                                                                  suppress_logging = suppress_logging))
                    
    
                # Get time traces
                
                if is_cross_corr:
                    # Two distinct channels
                    time_trace_counts_1, time_trace_t = self.get_time_trace(channels_spec_norm_ch1,
                                                                            time_trace_sampling,
                                                                            calling_function = 'run_standard_pipeline',
                                                                            suppress_logging = suppress_logging)
                    
                    time_trace_counts_2, _ = self.get_time_trace(channels_spec_norm_ch2,
                                                                 time_trace_sampling,
                                                                 calling_function = 'run_standard_pipeline',
                                                                 suppress_logging = suppress_logging)
                    
                    # Concatenate for further processing
                    for_reshape = (time_trace_counts_1.shape[0], 1)
                    time_traces = np.concatenate((time_trace_counts_1.reshape(for_reshape), time_trace_counts_2.reshape(for_reshape)), axis = 1)
                    
                else:
                    # Single channel
                    time_trace_counts, time_trace_t = self.get_time_trace(channels_spec_norm_ch1,
                                                                          time_trace_sampling,
                                                                          calling_function = 'run_standard_pipeline',
                                                                          suppress_logging = suppress_logging)
                    
                    # We crete a dummy second dimension, although this is not even strictly required
                    time_traces = time_trace_counts.reshape((time_trace_counts.shape[0], 1))
                    
                # Run actual burst removal
                _ = self.run_burst_removal(time_traces, 
                                           time_trace_sampling,
                                           calling_function = 'run_standard_pipeline',
                                           suppress_logging = suppress_logging)
                
                if write_intermediate_ccs or not (use_drift_correction or use_flcs_bg_corr or use_mse_filter):
                    # Correlate with filters up to this point applied
                    _ = self.get_correlation_uncertainty(channels_spec_norm_ch1,
                                                         channels_spec_norm_ch2, 
                                                         use_drift_correction = False,
                                                         use_flcs_bg_corr = False,
                                                         use_burst_removal = use_burst_removal,
                                                         use_mse_filter = False,
                                                         default_uncertainty_method = default_uncertainty_method,
                                                         calling_function = 'run_standard_pipeline',
                                                         suppress_logging = suppress_logging)
                
            except:
                # Crash during burst removal
                if not suppress_logging:
                    self.write_to_logfile(log_message = 'Default pipeline crashed in burst removal. Trying to calculate and write correlation fucntion without filters.',
                                          calling_function = 'run_standard_pipeline')
                    
                # Correlate with filters up to this point applied
                _ = self.get_correlation_uncertainty(channels_spec_norm_ch1,
                                                     channels_spec_norm_ch2, 
                                                     use_drift_correction = False,
                                                     use_flcs_bg_corr = False,
                                                     use_burst_removal = False,
                                                     use_mse_filter = False,
                                                     default_uncertainty_method = default_uncertainty_method,
                                                     calling_function = 'run_standard_pipeline',
                                                     suppress_logging = suppress_logging)

        # END of burst removal block
        
        # Second filter: bleaching/drift correction
        if use_drift_correction:
            
            try:
                # Get time trace
                time_trace_sampling = self.get_trace_time_scale(channels_spec_norm_ch1,
                                                                use_burst_removal = use_burst_removal,
                                                                calling_function = 'run_standard_pipeline')
                time_trace_counts, time_trace_t = self.get_time_trace(channels_spec_norm_ch1,
                                                                      time_trace_sampling,
                                                                      use_burst_removal = use_burst_removal,
                                                                      calling_function = 'run_standard_pipeline',
                                                                      suppress_logging = suppress_logging)
                
                # Run drift/bleaching correction
                _ = self.polynomial_undrifting_rss(time_trace_counts, 
                                                   time_trace_t, 
                                                   channels_spec_norm_ch1,
                                                   use_burst_removal = use_burst_removal,
                                                   calling_function = 'run_standard_pipeline',
                                                   suppress_logging = suppress_logging)
                
                if is_cross_corr:
                    # Second channel
                    
                    # Get time trace
                    time_trace_sampling = self.get_trace_time_scale(channels_spec_norm_ch2,
                                                                    use_burst_removal = use_burst_removal,
                                                                    calling_function = 'run_standard_pipeline',
                                                                    suppress_logging = suppress_logging)
                    time_trace_counts, time_trace_t = self.get_time_trace(channels_spec_norm_ch2,
                                                                          time_trace_sampling,
                                                                          use_burst_removal = use_burst_removal,
                                                                          calling_function = 'run_standard_pipeline',
                                                                          suppress_logging = suppress_logging)
                    
                    # Run drift/bleaching correction
                    _ = self.polynomial_undrifting_rss(time_trace_counts, 
                                                       time_trace_t, 
                                                       channels_spec_norm_ch2,
                                                       use_burst_removal = use_burst_removal,
                                                       calling_function = 'run_standard_pipeline',
                                                       suppress_logging = suppress_logging)
    
                if write_intermediate_ccs or not (use_flcs_bg_corr or use_mse_filter):
                    # Correlate with filters up to this point applied
                    _ = self.get_correlation_uncertainty(channels_spec_norm_ch1,
                                                         channels_spec_norm_ch2, 
                                                         use_drift_correction = use_drift_correction,
                                                         use_flcs_bg_corr = False,
                                                         use_burst_removal = use_burst_removal,
                                                         use_mse_filter = False,
                                                         default_uncertainty_method = default_uncertainty_method,
                                                         calling_function = 'run_standard_pipeline',
                                                         suppress_logging = suppress_logging)

            except:
                # Crash during drift/bleaching correction
                if not suppress_logging:
                    self.write_to_logfile(log_message = 'Default pipeline crashed in drift/bleaching correction. Trying to calculate and write correlation function with filters calculated so far.',
                                          calling_function = 'run_standard_pipeline')
                    
                # Correlate with filters up to this point applied
                _ = self.get_correlation_uncertainty(channels_spec_norm_ch1,
                                                     channels_spec_norm_ch2, 
                                                     use_drift_correction = False,
                                                     use_flcs_bg_corr = False,
                                                     use_burst_removal = use_burst_removal,
                                                     use_mse_filter = False,
                                                     default_uncertainty_method = default_uncertainty_method,
                                                     calling_function = 'run_standard_pipeline',
                                                     suppress_logging = suppress_logging)

        # END of bleaching/drift correction block
        
        # Third filter: Removal of anomalous segments based on mse between correlation functions
        if use_mse_filter:
            
            try:
                # Running this filter is a single function call, whether you have one or two channels
                _ = self.run_mse_filter(channels_spec_norm_ch1, 
                                        channels_spec_norm_ch2,
                                        use_drift_correction = use_drift_correction,
                                        use_burst_removal = use_burst_removal,
                                        calling_function = 'run_standard_pipeline',
                                        suppress_logging = suppress_logging)
                
                if write_intermediate_ccs or not use_flcs_bg_corr:

                    # Correlate with filters up to this point applied
                    _ = self.get_correlation_uncertainty(channels_spec_norm_ch1,
                                                         channels_spec_norm_ch2, 
                                                         use_drift_correction = use_drift_correction,
                                                         use_flcs_bg_corr = False,
                                                         use_burst_removal = use_burst_removal,
                                                         use_mse_filter = use_mse_filter,
                                                         default_uncertainty_method = default_uncertainty_method,
                                                         calling_function = 'run_standard_pipeline',
                                                         suppress_logging = suppress_logging)
                
            except:
                # Crash during anomalous segments removal
                if not suppress_logging:
                    self.write_to_logfile(log_message = 'Default pipeline crashed in anomalous segments removal. Trying to calculate and write correlation function with filters calculated so far.',
                                          calling_function = 'run_standard_pipeline')
                    
                # Correlate with filters up to this point applied
                _ = self.get_correlation_uncertainty(channels_spec_norm_ch1,
                                                     channels_spec_norm_ch2, 
                                                     use_drift_correction = use_drift_correction,
                                                     use_flcs_bg_corr = False,
                                                     use_burst_removal = use_burst_removal,
                                                     use_mse_filter = False,
                                                     default_uncertainty_method = default_uncertainty_method,
                                                     calling_function = 'run_standard_pipeline',
                                                     suppress_logging = suppress_logging)

        # END of anomalous segment removal block
        
        # Fourth filter: FLCS background subtraction
        if use_flcs_bg_corr:
            
            try:
                # Get TCSPC histogram
                tcspc_x, tcspc_y = self.get_tcspc_histogram(channels_spec_norm_ch1,
                                                            use_drift_correction = use_drift_correction,
                                                            use_burst_removal = use_burst_removal,
                                                            use_mse_filter = use_mse_filter,
                                                            calling_function = 'run_standard_pipeline',
                                                            suppress_logging = suppress_logging)
                
                # Find suitable range for tail fitting, and perform tail fit (we simply start 2 ns after peak)
                peak_position = np.argmax(tcspc_y)
                fit_start = np.uint64(peak_position + np.ceil(2. / self._micro_time_resolution))
                flat_background, _ = self.get_background_tail_fit(channels_spec_norm_ch1, 
                                                                  peak_position, 
                                                                  fit_start,
                                                                  use_drift_correction = use_drift_correction,
                                                                  use_burst_removal = use_burst_removal,
                                                                  use_mse_filter = use_mse_filter,
                                                                  calling_function = 'run_standard_pipeline',
                                                                  suppress_logging = suppress_logging)
                
                # Get FLCS weights
                _ = self.get_flcs_background_filter(tcspc_x, 
                                                    tcspc_y, 
                                                    flat_background, 
                                                    channels_spec_norm_ch1,
                                                    calling_function = 'run_standard_pipeline',
                                                    suppress_logging = suppress_logging)
                
                if is_cross_corr:
                    # Second channel
                    
                    # Get TCSPC histogram
                    tcspc_x, tcspc_y = self.get_tcspc_histogram(channels_spec_norm_ch2,
                                                                use_drift_correction = use_drift_correction,
                                                                use_burst_removal = use_burst_removal,
                                                                use_mse_filter = use_mse_filter,
                                                                calling_function = 'run_standard_pipeline',
                                                                suppress_logging = suppress_logging)
                    
                    # Find suitable range for tail fitting, and perform tail fit
                    peak_position = np.argmax(tcspc_y)
                    fit_start = np.uint64(peak_position + np.ceil(2. / self._micro_time_resolution))
                    flat_background, _ = self.get_background_tail_fit(channels_spec_norm_ch2, 
                                                                      peak_position, 
                                                                      fit_start,
                                                                      use_drift_correction = use_drift_correction,
                                                                      use_burst_removal = use_burst_removal,
                                                                      use_mse_filter = use_mse_filter,
                                                                      calling_function = 'run_standard_pipeline',
                                                                      suppress_logging = suppress_logging)
                    
                    # Get FLCS weights
                    _ = self.get_flcs_background_filter(tcspc_x, 
                                                        tcspc_y, 
                                                        flat_background, 
                                                        channels_spec_norm_ch2,
                                                        calling_function = 'run_standard_pipeline',
                                                        suppress_logging = suppress_logging)
                
                # Correlate with filters up to this point applied
                _ = self.get_correlation_uncertainty(channels_spec_norm_ch1,
                                                     channels_spec_norm_ch2, 
                                                     use_drift_correction = use_drift_correction,
                                                     use_flcs_bg_corr = use_flcs_bg_corr,
                                                     use_burst_removal = use_burst_removal,
                                                     use_mse_filter = use_mse_filter,
                                                     default_uncertainty_method = default_uncertainty_method,
                                                     calling_function = 'run_standard_pipeline',
                                                     suppress_logging = suppress_logging)

                
            except:
                # Crash during FLCS background subtraction
                if not suppress_logging:
                    self.write_to_logfile(log_message = 'Default pipeline crashed in FLCS background subtraction. Trying to calculate and write correlation function with filters calculated so far.',
                                          calling_function = 'run_standard_pipeline')
                    
                # Correlate with filters up to this point applied
                _ = self.get_correlation_uncertainty(channels_spec_norm_ch1,
                                                     channels_spec_norm_ch2, 
                                                     use_drift_correction = use_drift_correction,
                                                     use_flcs_bg_corr = False,
                                                     use_burst_removal = use_burst_removal,
                                                     use_mse_filter = use_mse_filter,
                                                     default_uncertainty_method = default_uncertainty_method,
                                                     calling_function = 'run_standard_pipeline',
                                                     suppress_logging = suppress_logging)

            # END of background correction block
            


        if write_pcmh:
            # Get PCMH as well
            _ = self.get_PCMH(channels_spec_norm_ch1,
                              spacing = np.sqrt(2.),
                              normalize = False,
                              use_burst_removal = use_burst_removal,
                              use_mse_filter = use_mse_filter,
                              more_channels_specs = None,
                              suppress_logging = suppress_logging,
                              calling_function = 'run_standard_pipeline'
                              )
    
            if is_cross_corr:
                # Second channel
                _ = self.get_PCMH(channels_spec_norm_ch2,
                                  spacing = np.sqrt(2.),
                                  normalize = False,
                                  use_burst_removal = use_burst_removal,
                                  use_mse_filter = use_mse_filter,
                                  more_channels_specs = None,
                                  suppress_logging = suppress_logging,
                                  calling_function = 'run_standard_pipeline'
                                  )
                
                # Also construct sum pcmh output for cross-correlation channel
                try:
                    # Can be unsafe, although it should work with the way the pipeline is set up
                    _ = self.get_PCMH(channels_spec_norm_ch1,
                                      spacing = np.sqrt(2.),
                                      normalize = False,
                                      use_burst_removal = use_burst_removal,
                                      use_mse_filter = use_mse_filter,
                                      more_channels_specs = [channels_spec_norm_ch2],
                                      suppress_logging = suppress_logging,
                                      calling_function = 'run_standard_pipeline'
                                      )
                except:
                    # Crash during FLCS background subtraction
                    if not suppress_logging:
                        self.write_to_logfile(log_message = 'Could not construct sum channel PC(M)H. Logging traceback:' + traceback.format_exc(),
                                              calling_function = 'run_standard_pipeline')

