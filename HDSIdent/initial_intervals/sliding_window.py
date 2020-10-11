import pandas as pd
import numpy as np
from collections import defaultdict

class SlidingWindow(object):
    """
    """
    
    def __init__(self,
                 window_size, 
                 H_v, 
                 min_input_coupling=1,
                 min_output_coupling=1,
                 num_previous_indexes=0,
                 min_interval_length=None,
                 n_jobs=-1, 
                 verbose=0):

        self.window_size = window_size
        self.H_v = H_v
        self.min_input_coupling = min_input_coupling
        self.min_output_coupling = min_output_coupling
        self.num_previous_indexes = num_previous_indexes
        self.min_interval_length = min_interval_length
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

    def _initialize_internal_variables(self):
        """
        This function initializes the interval variables.
        """
        self.unified_intervals = defaultdict(list)
        self.intervals = defaultdict(list)
        self.var_arr = defaultdict(list)
        self.num_previous_elements = self.window_size//2
        self.num_forward_elements = self.window_size-self.num_previous_elements-1
        self._indicating_sequences = defaultdict(lambda: defaultdict(dict))
        self.sequential_indicating_sequences = defaultdict(dict)
        self.global_sequential_indicating_sequence = None
        self.unified_indicating_sequence = None
        
        if np.ndim(self.H_v) == 0:
            self.H_v = [self.H_v]
        
    def _sliding_window(self, data, data_cols, data_type):
        """
        This function applies a variance sliding window to the provided
        signal.
        """
        for data_idx in range(0,data.shape[1]):
            
            if data_cols is not None:
                data_idx_name = data_cols[data_idx]
            else:
                if data_type == 'input':
                    data_idx_name = 'input'+'_'+str(data_idx)
                else:
                    data_idx_name = 'output'+'_'+str(data_idx)
            
            for idx in range(0,len(data[:,data_idx])):
                if idx < self.num_previous_elements:
                    window_val = np.var(data[:idx+self.num_forward_elements+1,data_idx])
                    self.var_arr[data_idx_name].append(window_val)
                else:
                    window_val = np.var(data[idx-self.num_previous_elements:idx+self.num_forward_elements+1,data_idx])
                    self.var_arr[data_idx_name].append(window_val)
    
    def _define_indicating_sequences(self, data, data_cols, data_type):
        """
        This function creates an indicating sequence, i.e., an array containing 1's
        in the intervals of interest and 0's otherwise, for every input and output
        signal (which are defined by the data_type variable).
        
        For this class, a value of 1 is defined when the sliding window variance is
        greater than its threshold H_v.
        """
        for data_idx in range(0,data.shape[1]):
            
            if data_cols is not None:
                data_idx_name = data_cols[data_idx]
            else:
                if data_type == 'input':
                    data_idx_name = 'input'+'_'+str(data_idx)
                else:
                    data_idx_name = 'output'+'_'+str(data_idx)
            
            if data_type == 'output':
                H_v_idx = len(self.H_v) - data.shape[1] + data_idx
            else:
                H_v_idx = data_idx
                
            indicating_idxs = np.where(np.array(self.var_arr[data_idx_name]) > self.H_v[H_v_idx])[0]
            self._indicating_sequences[data_type][data_idx_name] = np.zeros(len(data[:,data_idx]))
            self._indicating_sequences[data_type][data_idx_name][indicating_idxs] = 1  
        
    def _unify_indicating_sequences(self):
        """
        This function unifies the input and output indicating sequences.
        
        Output:
            unified_indicating_sequence: the unified indicatign sequences.
        """
        
        #Unify every input sequence
        unified_input_indicating_sequence = \
            np.array(np.array(list(self._indicating_sequences['input'].values()))).max(axis=0)
        
        #Unify every output sequence
        unified_output_indicating_sequence = \
            np.array(np.array(list(self._indicating_sequences['output'].values()))).max(axis=0)
        
        #Unify Input and Output
        unified_indicating_sequence = \
            np.array([unified_input_indicating_sequence,unified_output_indicating_sequence]).max(axis=0)
        
        return unified_indicating_sequence
    
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
    
    def _get_sequential_sequences(self, data, data_cols, data_type):
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
        
        for col_idx in range(data.shape[1]):
            
            if data_cols is not None:
                data_idx_name = data_cols[col_idx]
            else:
                data_idx_name = data_type+'_'+str(col_idx)
                
            self.sequential_indicating_sequences[data_type][data_idx_name] = \
                self._create_sequential_indicating_sequences(indicating_sequence=
                                                             self._indicating_sequences[data_type] \
                                                                                       [data_idx_name])
            
    def _get_final_intervals(self, global_sequential_indicating_sequence):
        """
        This function takes the global indicating sequences, i.e., the unified
        indicating sequence for all input and output signals and verfies if
        there is at least one input and one output valid indicating sequence inside
        each global indicating sequence.
        
        Arguments:
            global_sequential_indicating_sequence: the unified indicating sequence for
            all input and output signals.
        """
        
        final_segment_indexes = []
        
        for segment_idx_arr in global_sequential_indicating_sequence:
            
            #Check if at least one input indicating sequence is in the correspondig global sequence
            input_count = 0
            for input_name in self.sequential_indicating_sequences['input'].keys():
                input_aux_count = 0
                for input_sequence in self.sequential_indicating_sequences['input'][input_name]:
                    if all(elem in segment_idx_arr for elem in input_sequence):
                        input_aux_count+=1
                if input_aux_count > 0:
                    input_count += 1
                    
            #Check if at least one output indicating sequence is in the correspondig global sequence
            output_count = 0
            for output_name in self.sequential_indicating_sequences['output'].keys():
                output_aux_count = 0
                for output_sequence in self.sequential_indicating_sequences['output'][output_name]:
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
            - Applies the recursive exponential moving average/variance
            - Compute the initial intervals (change-points)
            - Creates an indicating sequence, unifying input and output intervals
            - From the indicating sequence, creates a final unified interval
        
        Output:
            unified_intervals: the final unified intervals for the input and output signals
        """
        
        #Verify data format
        X, y, X_cols, y_cols = self._verify_data(X,y)
        
        #Initialize Internal Variables
        self._initialize_internal_variables()
        
        #Apply Sliding Windows
        self._sliding_window(X,X_cols,'input')
        self._sliding_window(y,y_cols,'output')
        
        #Define Indicating Sequences
        self._define_indicating_sequences(X, X_cols, 'input')
        self._define_indicating_sequences(y, y_cols, 'output')
        
        #Unify Indicating Sequences From Inputs and Outputs
        self.unified_indicating_sequence = self._unify_indicating_sequences()
    
        #Get Sequential Sequences
        self._get_sequential_sequences(data=X, data_cols=X_cols, data_type='input')
        self._get_sequential_sequences(data=y, data_cols=y_cols, data_type='output')
        
        #Get Global Sequence
        self.global_sequential_indicating_sequence = \
            self._create_sequential_indicating_sequences(indicating_sequence=
                                                         self.unified_indicating_sequence)
        
        #Get Final Interval Respecting min_input_coupling and min_output_coupling
        final_segment_indexes = self._get_final_intervals(self.global_sequential_indicating_sequence)
        
        self.unified_intervals = dict(zip(range(0,len(final_segment_indexes)),
                                          final_segment_indexes))
        
        #Length Check
        if ((self.min_interval_length is not None) and 
            (self.min_interval_length > 1)):
            self.unified_intervals = self._length_check()

        return self.unified_intervals