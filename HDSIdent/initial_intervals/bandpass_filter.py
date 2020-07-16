import pandas as pd
import numpy as np
from scipy import signal
from collections import defaultdict
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

class BandpassFilter(object):
    """
    Performs signal segmentation using a discrete-time Butterworth
    bandpass filter from SciPy. Notice that the input frequencies are
    normalized into 0 and 1.
    """
    def __init__(self,
                 W, 
                 N, 
                 sigma = None, 
                 H=None, 
                 min_input_coupling=1,
                 min_output_coupling=1,
                 num_previous_indexes=0,
                 min_interval_length=None,
                 n_jobs=-1, 
                 verbose=0):
        """ 
        Constructor.

        Arguments:
            X: the input discrete-time data
            W: input frequency [W1, W2] array as in scipy documentation
            N: Butterworth filter order
            sigma: data (population) standard deviation
            H: change-point threshold
            n_jobs: the number of CPUs to use
            verbose: the degree of verbosity (going from 0 to 10)
        """
        self.W = W
        self.N = N
        self.sigma = sigma
        self.H = H
        self.min_input_coupling=min_input_coupling
        self.min_output_coupling=min_output_coupling
        self.num_previous_indexes=num_previous_indexes
        self.min_interval_length=min_interval_length
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def _verify_data(self,X,y):
        """
        Verifies the data type and save data columns
        in case they are provided.
        
        Arguments:
            X: the input data in pandas dataframe format or numpy array
            y: the output data in pandas dataframe format or numpy array
        
        Output:
            X: the input data in numpy array format
            y: the input data in numpy array format
            X_cols: the input data columns in case they are provided
            y_cols: the output data columns in case they are provided
        """
        if type(X) == pd.core.frame.DataFrame:
            X_cols = X.columns
            X = X.values
            if X.ndim == 1:
                X = X.reshape(-1,1)
        elif type(X) == np.ndarray:
            X_cols = None
            if X.ndim == 1:
                X = X.reshape(-1,1)
        else:
            raise Exception("Input data must be a pandas dataframe or a numpy array") 

        if type(y) == pd.core.frame.DataFrame:
            y_cols = y.columns
            y = y.values
            if y.ndim == 1:
                y = y.reshape(-1,1)
        elif type(y) == np.ndarray:
            y_cols = None
            if y.ndim == 1:
                y = y.reshape(-1,1)
        else:
            raise Exception("Input data must be a pandas dataframe or a numpy array") 
            
        return X, y, X_cols, y_cols
    
    def _initialize_internal_variables(self, X):
        """
        THis function initializes the required
        internal variables.
        """
        self._indicating_sequences = defaultdict(lambda: defaultdict(dict))
        self.sequential_indicating_sequences = defaultdict(dict)
        self.global_sequential_indicating_sequence = None
        self.unified_intervals = None
        self.intervals = None
        self.butt_mtrx = None
        self._is_interval = [False]*X.shape[1]
        self._init_idx = [0]*X.shape[1]
        self._final_idx = [0]*X.shape[1]
        self._err_points = list()
        self._num = 0
        self._den = 0

        if self.sigma is None:
            self.sigma = np.std(X,axis=0)
        
        if self.H is None:
            self.H = 5*self.sigma
        
        if self.N is None:
            self.N = 1
        
    def butterworth_filter(self, X):
        """
        Apply a Butterworth bandpass filter to the input data.
        
        Output:
            butt_mtrx: the filtered data
        """
        
        #Create filtered signal array
        self.butt_mtrx = np.empty(shape=(X.shape[0],X.shape[1]))
        
        #Define analog filter
        self.num, self.den = signal.butter(N = self.N, 
                                           Wn = self.W, 
                                           btype='bandpass', 
                                           analog=True)
        
        #Compute transfer function
        e = signal.TransferFunction(self.num, self.den)
        
        #Filter each signal (column)
        for col in range(X.shape[1]):
            
            #Input initial signal to avoid deflection
            X_aux = [X[0,col]]*10000+list(X[:,col])
            
            #Filter
            t_in = np.arange(0,len(X_aux),1)
            t_out, butt_arr, _ = signal.lsim(e, X_aux, t_in)
            self.butt_mtrx[:,col] = butt_arr.reshape(-1,1)[10000:,0] 
    
    def _define_deviations_from_the_mean(self, X):
        """
        Deviation indexes are those in which the absolute
        value of the filtered signal is higher then a given
        threshold H.
        """
        
        #Compute deviations for each signal (column)
        for col in range(X.shape[1]):
            indicating_idxs = np.where(np.abs(self.butt_mtrx[:,col]) >= self.H[col])[0]
            self._indicating_sequences[col] = np.zeros(len(X[:,col]))
            self._indicating_sequences[col][indicating_idxs] = 1      
                
    def _unify_indicating_sequences(self):
        """
        The resulting indicating sequences are unified to
        obtain a single interval. Let us call the indicating
        sequences for the input Iu and the indicating sequences
        for the output Iy. The unified indicating sequence is
        the defined as Iu U Iy.
        """
        
        indicating_sequence = np.array(self._indicating_sequences[0])
        for key, value in self._indicating_sequences.items():
            if key > 0:
                indicating_sequence = np.maximum(indicating_sequence,
                                                 np.array(value))
        return indicating_sequence        
        
    def _create_sequential_indicating_sequences(self, indicating_sequence):
        """
        This function gets the indicating sequence for a given data
        and creates the corresponding segments where the sequence 
        contains consecutive values of 1. For example, the sequence
        [0,0,1,1,1,1,0,0,0,1,1,0,0,0] would result in two sequential
        sequences:
        
        1) Sequence formed by indexes [2,3,4,5]
        2) Sequence forme by indexes [9,10]
        
        Arguments:
            indicating_sequence: the data indicating sequence.
        
        Output:
            sequential_indicating_sequences: the sequential indicating sequence.
        """
        
        is_interval = False
        sequential_indicating_sequences = []
        aux_arr = []
        
        for idx in range(len(indicating_sequence)):

            if not is_interval and indicating_sequence[idx] == 1:
                is_interval = True

            if is_interval and indicating_sequence[idx] == 1:
                aux_arr.append(idx)

            if idx < len(indicating_sequence)-1:
                if (is_interval 
                    and indicating_sequence[idx] == 1
                    and indicating_sequence[idx+1] == 0):

                    is_interval = False
                    sequential_indicating_sequences.append(aux_arr)
                    aux_arr = []
            else:
                if aux_arr != []:
                    sequential_indicating_sequences.append(aux_arr)
                    
        return sequential_indicating_sequences
    
    def _get_sequential_sequences(self, X, data_cols, input_size):
        """
        This function gets the indicating sequence for a given data
        and creates the corresponding segments where the sequence 
        contains consecutive values of 1. For example, the sequence
        [0,0,1,1,1,1,0,0,0,1,1,0,0,0] would result in two sequential
        sequences:
        
        1) Sequence formed by indexes [2,3,4,5]
        2) Sequence forme by indexes [9,10]
        
        Arguments:
            data: a data matrix (either input or output data)
            data_cols: the columns names of the data matrix
            data_type: the data type (input or output)
        """
        name_idx = 0
        for col_idx in range(X.shape[1]):
            
            if col_idx <= input_size-1:
                data_type = 'input'
                name_idx = col_idx
            else:
                data_type = 'output'
                name_idx = col_idx - input_size
                
            if data_cols is not None:
                data_idx_name = data_cols[col_idx]
            else:
                data_idx_name = data_type+'_'+str(name_idx)
                
            self.sequential_indicating_sequences[data_type][data_idx_name] = \
                self._create_sequential_indicating_sequences(indicating_sequence=
                                                             self._indicating_sequences[col_idx])
        return self.sequential_indicating_sequences
    
    def _extend_previous_indexes(self):
        """
        This function allows an extension of each interval
        with previous index values. The number of indexes 
        extended are provided in num_previous_indexes.
        """
        for key_1, dict_1 in self.sequential_indicating_sequences.items():
            for key_2, interval_arr in dict_1.items():
                for idx, interval in enumerate(interval_arr):

                    min_val = np.min(interval)

                    if ((idx == 0) and 
                        (np.min(interval)-self.num_previous_indexes < 0)
                       ):
                        min_val = 0
                    elif ((idx > 0) and 
                          ((np.min(interval)-self.num_previous_indexes) <= np.max(interval_arr[idx-1]))
                         ):
                        min_val = np.max(interval_arr[idx-1])+1
                    else:
                        min_val = np.min(interval)-self.num_previous_indexes

                    self.sequential_indicating_sequences[key_1][key_2][idx] = list(range(min_val,
                                                                                         np.max(interval)+1))
                    
    def _update_indicating_sequences(self, X, data_cols, input_size):
        """
        This function is used when an _extend_previous_indexes is
        performed. If the sequential intervals are extended, the
        indicating sequences must be updated before they are unified.
        """
        name_idx = 0
        for col_idx in range(X.shape[1]):
            
            if col_idx <= input_size-1:
                data_type = 'input'
                name_idx = col_idx
            else:
                data_type = 'output'
                name_idx = col_idx - input_size
                
            if data_cols is not None:
                data_idx_name = data_cols[col_idx]
            else:
                data_idx_name = data_type+'_'+str(name_idx)
            
            self._indicating_sequences[col_idx] = np.zeros(len(X[:,col_idx]))
            sequential_seq = self.sequential_indicating_sequences[data_type][data_idx_name]
            
            for seq in sequential_seq:
                self._indicating_sequences[col_idx][seq] = 1
                
    def _get_final_intervals(self, labeled_intervals, global_sequence):
        """
        This function takes the global indicating sequences, i.e., the unified
        indicating sequence for all input and output signals and verfies if
        there is at least one input and one output valid indicating sequence inside
        each global indicating sequence.
        
        Arguments:
            global_sequence: the unified intervals for all input and output signals.
            labeled_intervals: the individual intervals for each input and output.
        """
        
        final_segment_indexes = []
        
        for segment_idx_arr in global_sequence:
            
            #Check if at least one input indicating sequence is in the correspondig global sequence
            input_count = 0
            for input_name in labeled_intervals['input'].keys():
                input_aux_count = 0
                for input_sequence in labeled_intervals['input'][input_name]:
                    if all(elem in segment_idx_arr for elem in input_sequence):
                        input_aux_count+=1
                if input_aux_count > 0:
                    input_count += 1
                    
            #Check if at least one output indicating sequence is in the correspondig global sequence
            output_count = 0
            for output_name in labeled_intervals['output'].keys():
                output_aux_count = 0
                for output_sequence in labeled_intervals['output'][output_name]:
                    if all(elem in segment_idx_arr for elem in output_sequence):
                        output_aux_count+=1
                if output_aux_count > 0:
                    output_count += 1

            if (input_count >= self.min_input_coupling and 
                output_count >= self.min_output_coupling):
                
                final_segment_indexes.append(segment_idx_arr)
        
        return final_segment_indexes 
    
    def _length_check(self):
        """
        This function checks the interval length
        according to the provided min_interval_length.
        Only intervals with length >= min_interval_length
        are returned.
        """
        final_intervals = {}

        for key, value in self.unified_intervals.items():
            if len(value) >= self.min_interval_length:
                final_intervals[key] = value
        
        return final_intervals
    
    def fit(self, X, y):
        """
        This function performs the following routines:
            - Applies the Butterworth Filter in the signal
            - From the filtered signal, defines the initial intervals (change-points)
            - Creates an indicating sequence, unifying input and output intervals
            - From the indicating sequence, creates a final unified interval
        """
        #Verify data format
        X, y, X_cols, y_cols = self._verify_data(X,y)
        
        if ((X_cols is not None) and (y_cols is not None)):
            data_cols = list(X_cols)+list(y_cols)
        else:
            data_cols = None
            
        #Create Matrix
        data = np.concatenate([X,y],axis=1)
        
        #Initialize Internal Variables
        self._initialize_internal_variables(X=data)
        
        #Apply Butterworth Filter
        self.butterworth_filter(X=data)
        
        #Compute Deviations from the mean
        self._define_deviations_from_the_mean(X=data)
        
        #Compute Sequential Sequences for Each Signal
        sequential_sequences = \
        self._get_sequential_sequences(X=data, 
                                       data_cols=data_cols, 
                                       input_size=X.shape[1])
        
        #Extend Intervals
        if self.num_previous_indexes > 0:
            self._extend_previous_indexes()
            self._update_indicating_sequences(X=data,
                                              data_cols=data_cols,
                                              input_size=X.shape[1])
            
        #Unify Indicating Sequences
        self.unified_indicating_sequence = self._unify_indicating_sequences()
        
        #Get Global Sequential Sequence (Unified Sequence)
        self.global_sequence = \
            self._create_sequential_indicating_sequences(indicating_sequence=
                                                         self.unified_indicating_sequence)
        
        #Find intervals that respect min_input_coupling and min_output_coupling
        final_segment_indexes = self._get_final_intervals(labeled_intervals=sequential_sequences, 
                                                          global_sequence=self.global_sequence)
        
        self.unified_intervals = dict(zip(range(0,len(final_segment_indexes)),
                                          final_segment_indexes))
        
        #Length Check
        if ((self.min_interval_length is not None) and 
            (self.min_interval_length > 1)):
            self.unified_intervals = self._length_check()
            
        return self.unified_intervals
    
    def plot_change_points(self, X, y, threshold_name='H'):
        """
        Plots all found change points and its corresponding
        intervals.
        """
        #Verify data format
        X, y, X_cols, y_cols = self._verify_data(X,y)
        
        if ((X_cols is not None) and (y_cols is not None)):
            df_cols = list(X_cols)+list(y_cols)
        else:
            df_cols = None
            
        #Create Matrix
        data = np.concatenate([X,y],axis=1)
        
        #Verify if fit was performed
        try:
            self.unified_intervals
        except:
            self.fit(X=X, y=y)
        
        #Make plot
        for col in range(len(self._indicating_sequences.keys())):
            
            #Take deviation from the mean for current signal
            deviation_idxs = np.argwhere(self._indicating_sequences[col]==1)
            
            #Plot thresholds
            X = data
            H = self.H
            
            sns.set_style("darkgrid")
            plt.figure(figsize=(15,5))
            
            #Plot filtered signal
            plt.plot(self.butt_mtrx[:,col], color='coral', linewidth=0.8, zorder=1)
            
            plt.plot([H[col]]*len(X),color='black',linestyle='--')
            plt.annotate("+{} = {}".format(threshold_name,H[col]),
                         xy=(10*len(X)/10.8, np.max(self.butt_mtrx[:,col])*0.4),
                         fontsize=20,
                         fontweight='bold',
                         color='black')
            
            plt.plot([-H[col]]*len(X),color='black',linestyle='--',label=r"$l_{e}$ Threshold")
            plt.annotate("-{} = {}".format(threshold_name,-H[col]),
                         xy=(10*len(X)/10.8, -np.max(self.butt_mtrx[:,col])*0.45),
                         fontsize=20,
                         fontweight='bold',
                         color='black')
            
            if df_cols is None:
                col_name = f"Signal {col}"
            else:
                col_name = f"Signal {df_cols[col]}"
                
            plt.title(f"Bandpass Filter Change Points and Intervals for {col_name}", fontsize=20, fontweight='bold')
            plt.ylabel("Signal Amplitude", fontsize=20)
            plt.xlabel("Discrete Samples", fontsize=20)
            plt.xticks(fontsize=20, color='black')
            plt.yticks(fontsize=20, color='black')
            
            #Plot deviation from the mean
            plt.scatter(deviation_idxs, 
                        self.butt_mtrx[:,col][deviation_idxs], 
                        s=0.5, 
                        color='darkred',
                        zorder=2,
                        label="Intervals")
            
            plt.legend(fontsize=18,markerscale=10)
            plt.show()