from joblib import Parallel, delayed
from collections import defaultdict
from scipy import stats
from copy import deepcopy
import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from collections import defaultdict
from scipy import stats
from copy import deepcopy
import pandas as pd
import numpy as np

class MIMOStatistical(object):
    """
    This class performs a multivariable signal segmentation
    for System Identification. The statistical method implemented
    in this class was based on the following reference:
    
    WANG, J. et al. Searching historical data segments for process
    identification in feedback control loops. Computers and Chemical
    Engineering, v. 112, n. 6, p. 6–16, 2018.
    
    The following steps are performed by this class:
    
    1) Initial segments are provided for analysis;
    
    2) For each segment and for each signal, verify if
    the provided segment suffers a fair amount of magnitude
    changes. For this, a non parametric kolmogorov-smirnov
    test is performed;
    
    3) For each segment and for each signal, verify if two
    consecutive intervals contain different mean values. A
    t-student for unknown mean and variance is performed for
    this test;
    
    4) Unify input and output data that meet criteria 1 to 3;
    
    5) Return the intervals from step 4 that contain at least
    one input and one output satisfying 1 to 3.
    
    Arguments:
        initial_intervals: the initial intervals (segmentation) for 
        each input and output signal;
        ks_p_value: the p-value threshold for the kolmogorov-smirnov test;
        ts_p_value: the p-value threshold for the t-student test;
        compare_means: whether or not perform the t-student test;
        min_input_coupling: the min number of inputs that must satify the statistical tests;
        min_output_coupling: the min number of outputs that must satisfy the statistical tests;
        mean_crossing_percentual: The percentage over the size of an interval
        that serves as a threshold to determine the minimum number of mean-crossings 
        required to perform the non parametric kolmogorov-smirnov test;
        noise_std: the noise standard deviation that is included in each segment for
        performing the non parametric kolmogorov-smirnov test;
        verbose: the degree of verbosity from 0 to 10, being 0 the absence of verbose.
        n_jobs: the number off CPU's to use.
        
    """
    def __init__(self,
                 initial_intervals,
                 ks_p_value=1e-20,
                 ts_p_value=1e-50,
                 compare_means=True,
                 min_input_coupling=1,
                 min_output_coupling=1,
                 mean_crossing_percentual=0.5,
                 noise_std=0.05,
                 verbose=0,
                 n_jobs=-1):
        
        self.initial_intervals = initial_intervals
        self.ks_p_value = ks_p_value
        self.ts_p_value = ts_p_value
        self.compare_means = compare_means
        self.min_input_coupling = min_input_coupling
        self.min_output_coupling = min_output_coupling
        self.mean_crossing_percentual = mean_crossing_percentual
        self.noise_std = noise_std
        self.verbose = verbose
        self.n_jobs=n_jobs

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
    
    def _initialize_indicating_sequences(self, X, X_cols, y, y_cols):
        """
        This function initializes the indicating sequences. An
        indicating sequence is an array of 0's and 1's. The size
        of an indicating sequence is equal to number of initial
        intervals provided. The array value is 1 where an interval 
        meets a certain criteria and 0 otherwise.
        
        Arguments:
            X: the input matrix. Each column corresponds to an input signal
            X_cols: the input signals column names
            y: the output matrix: Each column corresponds to an ouput signal
            y_cols: the output signals column names
        """
        self.indicating_sequences = defaultdict(dict)
        
        for input_idx in range(0,X.shape[1]):
            
            if X_cols is not None:
                input_idx_name = X_cols[input_idx]
            else:
                input_idx_name = 'input'+'_'+str(input_idx)
            
            self.indicating_sequences['input'][input_idx_name] = \
                np.zeros(len(X))
        
        for output_idx in range(0,y.shape[1]):
            
            if y_cols is not None:
                output_idx_name = y_cols[output_idx]
            else:
                output_idx_name = 'output'+'_'+str(output_idx)
                
            self.indicating_sequences['output'][output_idx_name] = \
                np.zeros(len(y))


    def _insert_random_noise(self, data_segment, noise_mean, noise_std):
        """
        This function creates a random noise with a provided
        mean and standard deviation.
        
        Arguments:
            data_segment: a segment of data
            noise_mean: the noise mean
            noise_std: the noise standard deviation
        
        Output:
            data_noise: the original signal contaminated by noise
        """
        np.random.seed(0)
        noise = np.random.normal(noise_mean, noise_std, data_segment.shape)
        data_noise = data_segment + noise

        return data_noise

    def _find_mean_crossing_indexes(self, data_segment):
        """
        This function finds the mean crossing indexes
        of a given segment of data. Two consecutive data
        points constitute a mean crossing if they satisfy
        the following equation:
        
        (data_segment(tc)-avg(data_segment))*(data_segment(tc+1)-avg(data_segment)) <= 0
        
        The mean crossing point is considered the first of the
        two consecutive points, i.e., tc.
        
        Arguments:
            data_segment: a data segment
        
        Output:
            mean_crossing_idx_arr: an array with the mean crossing indexes
        """
        mean_crossing_idx_arr = []
        interval_avg = np.mean(data_segment)

        points_k = data_segment[:-1] - interval_avg
        points_k_1 = data_segment[1:] - interval_avg

        mean_crossing_idx_arr = np.squeeze(np.argwhere(points_k*points_k_1 <= 0))

        return mean_crossing_idx_arr

    def _create_mean_crossing_statistic(self, mean_crossing_idx_arr):
        """
        This function creates the mean-crossing statistic. This
        statistic is defined as the number of consecutive mean
        crossing values. If the number of consecutive mean crossing
        values follows a exponential distribution, it means that
        the signal does not experience significant magnitude 
        changes.
        
        Arguments:
            mean_crossing_idx_arr: the mean crossing indexes
        
        Output:
            Tc: the mean crossing statistic
        """

        if mean_crossing_idx_arr.size >= 2:
            Tc = mean_crossing_idx_arr[1:] - mean_crossing_idx_arr[:-1]
            Tc = np.unique(Tc, return_counts=True)[1]
        else:
            Tc = None

        return Tc

    def _kolmogorov_smirnov_test(self, Tc, num_mean_crossings, min_num_mean_crossings):
        """
        Computs a non-parametric Kolmogorov-Smirnov statistical
        test to check if the mean crossing statistic follows
        an exponential distribution. If it does, it mean that
        the data segment being considered does not experience a 
        significant amount of magnitude changes. On the contrary,
        on could reject the null hypothesis and consider that
        the segment experiences a fair amount of magnitude changes.
        
        Arguments:
            Tc: the mean crossing statistic.
            num_mean_crossings: the number of mean crossing experienced
            by the data segment.
            min_num_mean_crossings: the minimum number of mean crossings
            ir order to perfor the kolmogorov_smirnov teste. If the data
            segment has no mean crossing, it means it is constantly increasing
            or decreasing.
        
        Output:
            p_value: the Kolmogorov-Smirnov test p_value. If the p_value
            is lower then a given statistical significance level, then
            one can reject the null hypothesis that assumes Tc follows an
            exponential distribution.
        """

        if Tc is not None:
            if num_mean_crossings > min_num_mean_crossings:
                p_value = stats.kstest(Tc, cdf = 'expon', alternative = 'two-sided')[1]
            else:
                p_value = 0
        else:
            p_value = 0

        return p_value
    
    def _magnitude_changes_test(self, data_segment):
        """
        Performs all the steps for checking if a data
        segment experiences a fair amount of magnitude
        changes.
        
        Arguments:
            data_segment: a data segment.
        
        Output:
            p_value: the Kolmogorov-Smirnov test p_value. If the p_value
            is lower then a given statistical significance level, then
            one can reject the null hypothesis that assumes Tc follows an
            exponential distribution.   
        """
        
        #Insert Noise
        data_noise = self._insert_random_noise(data_segment=data_segment,
                                               noise_mean=0,
                                               noise_std=self.noise_std)

        #Fid Mean Crossing Indexes
        mean_crossing_idx_arr = self._find_mean_crossing_indexes(data_segment=data_noise)
        
        #Create Mean Crossing Statistic
        Tc = self._create_mean_crossing_statistic(mean_crossing_idx_arr=mean_crossing_idx_arr)

        #Compute Kolmogorov-Smirnov Statistical Test
        p_value = self._kolmogorov_smirnov_test(Tc=Tc,
                                                num_mean_crossings=len(mean_crossing_idx_arr),
                                                min_num_mean_crossings=self.mean_crossing_percentual*len(data_segment))

        return p_value
    
    def _fit_magnitude_changes(self, data, data_cols, data_type):
        """
        Given a data matrix (either a matrix of input data or a matrix
        of output data), this function fits the data according to the
        Kolmogorov-Smirnov test steps.
        
        Arguments:
            data: a data matrix (either input or output data)
            data_cols: the columns names of the data matrix
            data_type: the data type (input or output)
        """
        executor = Parallel(n_jobs=self.n_jobs,
                            verbose=self.verbose)
        
        for col_idx in range(0, data.shape[1]):
            
            if data_cols is not None:
                data_idx_name = data_cols[col_idx]
            else:
                data_idx_name = data_type+'_'+str(col_idx)
            
            #Perform Kolmogorov-Smirnov Test
            magnitude_change_task = (delayed(self._magnitude_changes_test)(data[segment, col_idx])
                                     for segment in self.data_segments_dict[data_type][data_idx_name])

            self.ks_p_values_dict[data_type][data_idx_name] = list(executor(magnitude_change_task))
            
            #Take Segment Indexes that Satifies the Hypothesis Test
            self.ks_segments_indexes[data_type][data_idx_name] = \
                np.squeeze(np.argwhere(np.array(self.ks_p_values_dict[data_type][data_idx_name]) < self.ks_p_value))
            
            #Update Indicating Sequences
            try:
                ks_segments_indexes = np.array(self.ks_segments_indexes[data_type][data_idx_name])
                data_signal_indexes = np.array(self.data_segments_dict[data_type][data_idx_name])   
                
                indicating_indexes = data_signal_indexes[ks_segments_indexes]
                
                if np.ndim(ks_segments_indexes) > 0:
                    indicating_indexes = np.concatenate(indicating_indexes, axis=0)
                
                self.indicating_sequences[data_type][data_idx_name][indicating_indexes] = 1
            except:
                pass
    
    def _difference_in_mean_test(self, data, segment_idx, data_type, data_idx_name, col_idx):
        """
        This function compares two consecutive intervals and test if they
        have a difference in mean. The null hypothesis is that both signals
        have the same mean value. The indicating sequence of
        each signal is updated case a particular segment meet the test
        specifications.
        
        A t-student with unknown mean and variance is performed to compare
        the intervals and the resulting p-value is compared to the provided
        significance threshold (ts_p_value).
        
        Arguments:
            data: a data matrix (either input or output data).
            segment_idx: a segment index. Notice that segment_idx + 1 is the following segment.
            data_type: the data type (input or output).
            data_idx_name: the column name (or index) for the given data.
            col_idx: the signal column that is being iterated.
        """
        
        #Create both intervals
        interval_1 = data[self.data_segments_dict[data_type][data_idx_name][segment_idx],col_idx]
        interval_2 = data[self.data_segments_dict[data_type][data_idx_name][segment_idx+1],col_idx]
        
        #Perform t-student test
        p_value = stats.ttest_ind(interval_1, interval_2, equal_var = False)[1]
        
        #Compare p-value agains threshold
        if p_value < self.ts_p_value:
            data_signal_indexes = np.array(self.data_segments_dict[data_type][data_idx_name])
            true_data_indexes = data_signal_indexes[segment_idx+1]
            self.indicating_sequences[data_type][data_idx_name][true_data_indexes] = 1.
            
        return p_value
    
    def _fit_difference_in_mean(self, data, data_cols, data_type):
        """
        This function performs all the steps required for testing
        the difference in mean of two consecutive intervals. All the
        intervals are tested sequentially. The indicating sequence of
        each signal is updated case a particular segment meet the test
        specifications.
        
        A t-student with unknown mean and variance is performed to compare
        the intervals and the resulting p-value is compared to the provided
        significance threshold (ts_p_value).
        
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

            executor = Parallel(n_jobs=self.n_jobs,
                                verbose=self.verbose,
                                require='sharedmem')
            
            difference_in_mean_task = (delayed(self._difference_in_mean_test)
                                      (data,segment_idx,data_type,data_idx_name,col_idx)
                                       for segment_idx in range(0,len(self.data_segments_dict[data_type] \
                                                                                             [data_idx_name])-1))
            
            self.ts_p_values_dict[data_type][data_idx_name] = list(executor(difference_in_mean_task))
        
    def _create_segments_dict(self, data, data_cols, data_type):
        """
        This function creates a data segment for each input and output
        signal based on the initial intervals provided.
        
        Arguments:
            data: a data matrix (either input or output data)
            data_cols: the columns names of the data matrix
            data_type: the data type (input or output)
        """
        for col in data_cols:
            self.data_segments_dict[data_type][col] = self.initial_intervals[col]  
    
    def _unify_indicating_sequences(self):
        """
        This function unifies the input and output indicating sequences.
        
        Output:
            unified_indicating_sequence: the unified indicatign sequences.
        """
        
        #Unify every input sequence
        unified_input_indicating_sequence = \
            np.array(np.array(list(self.indicating_sequences['input'].values()))).max(axis=0)
        
        #Unify every output sequence
        unified_output_indicating_sequence = \
            np.array(np.array(list(self.indicating_sequences['output'].values()))).max(axis=0)
        
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
        
        1) Sequence formed by indexes [1,2,3,5]
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
                                                             self.indicating_sequences[data_type] \
                                                                                      [data_idx_name])
            
    def _get_significant_intervals(self, global_sequential_indicating_sequence):
        """
        This function takes the global indicating sequences, i.e., the unified
        indicating sequence for all input and output signals and verfies if
        there is at least one input and one output valid indicating sequence inside
        each global indicating sequence.
        
        Arguments:
            global_sequential_indicating_sequence: the unified indicating sequence for
            all input and output signals.
        """
        
        significant_segment_indexes = []
        
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
                
                significant_segment_indexes.append(segment_idx_arr)
        
        return significant_segment_indexes 
        
    def fit(self, X, y):
        """
        This function performs all the steps required for
        perform the statistical segmentation of a multivariable
        system. The following steps are performed:
        
        1) Test for significant magnitude changes in the signal
        2) Test for significant differences in mean in the signal
        3) Unify input and output data
        
        Arguments:
            X: the input signal matrix. Each column corresponds to an
            input signal.
            y: the output signal matrix. Each column corresponds to an
            output signal.
        """
        #Internal Variables
        self.data_segments_dict = defaultdict(dict)
        self.ks_p_values_dict = defaultdict(dict)
        self.ks_segments_indexes = defaultdict(dict)
        self.ts_p_values_dict = defaultdict(dict)
        self.sequential_indicating_sequences = defaultdict(dict)
        self.global_sequential_indicating_sequence = None
        self.ks_indicating_sequences = None
        self.ts_indicating_sequences = None
        self.unified_indicating_sequence = None
        
        #Verify data format
        X, y, X_cols, y_cols = self._verify_data(X,y)
        
        #Initialize Indicating Sequences
        self._initialize_indicating_sequences(X=X, X_cols=X_cols, y=y, y_cols=y_cols)
        
        #Create Segments Dict
        self._create_segments_dict(data=X, data_cols=X_cols, data_type='input')
        self._create_segments_dict(data=y, data_cols=y_cols, data_type='output')
    
        #Magnitude Change Test
        self._fit_magnitude_changes(data=X, data_cols=X_cols, data_type='input')
        self._fit_magnitude_changes(data=y, data_cols=y_cols, data_type='output')
        self.ks_indicating_sequences = deepcopy(self.indicating_sequences)
        
        #Difference in Mean Test
        if self.compare_means:
            self._fit_difference_in_mean(data=X, data_cols=X_cols, data_type='input')
            self._fit_difference_in_mean(data=y, data_cols=y_cols, data_type='output')
            self.ts_indicating_sequences = deepcopy(self.indicating_sequences)
        
        #Unify Indicating Sequences From Inputs and Outputs
        self.unified_indicating_sequence = self._unify_indicating_sequences()
    
        #Sequential Indicating Sequences
        self._get_sequential_sequences(data=X, data_cols=X_cols, data_type='input')
        self._get_sequential_sequences(data=y, data_cols=y_cols, data_type='output')
        
        self.global_sequential_indicating_sequence = \
            self._create_sequential_indicating_sequences(indicating_sequence=
                                                         self.unified_indicating_sequence)
        
        #Select Intervals That Contains at Least one input and one output
        #satisfying the statistical criteria
        self.final_segments = self._get_significant_intervals(self.global_sequential_indicating_sequence)

        return self.final_segments