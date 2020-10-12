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
        each input and output signals;
        alpha: the significance level for the two-mean comparison t-student test;
        ks_critic: the critical value factor for the Kolmogorov-Smirnov (Lilliefors) test;
        insert_noise: whether or not to insert noise in the input signal;
        compare_means: whether or not to perform the t-student test;
        min_input_coupling: the min number of inputs that must satify the statistical tests;
        min_output_coupling: the min number of outputs that must satisfy the statistical tests;
        noise_std: the noise standard deviation (if inserted);
        verbose: the degree of verbosity from 0 to 10, being 0 the absence of verbose.
        n_jobs: the number of CPU's to be used.
        
    """
    def __init__(self,
                 initial_intervals,
                 alpha=0.01,
                 ks_critic = 1.25,
                 mean_delta = 0.05,
                 insert_noise_sp = False,
                 insert_noise_y = False,
                 compare_means=True,
                 min_input_coupling=1,
                 min_output_coupling=1,
                 noise_std=0.01,
                 verbose=0,
                 n_jobs=-1):
        
        self.initial_intervals = initial_intervals
        self.ks_critic = ks_critic
        self.alpha = alpha
        self.mean_delta = mean_delta
        self.insert_noise_sp = insert_noise_sp
        self.insert_noise_y = insert_noise_y
        self.compare_means = compare_means
        self.min_input_coupling = min_input_coupling
        self.min_output_coupling = min_output_coupling
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
    
    def _initialize_internal_variables(self):
        """
        This function initializes the following variables:
            - data_segments_dict: a dictionary with the data segments
            corresponding to the provided initial intervals;
            - ks_value_dict: a dictionary where the keys are the
            input/output signals and the values are the results
            from the lilliefors test for each segment (result values
            can be either 1 or 0);
            - ks_segments_indexes: the indexes of the segments where
            the lilliefors test successeded (i.e., returned 1);
            - ts_values_dict: a dictionary where the keys are the
            input/output signals and the values are the results
            from the t-student test for each segment (result values
            can be either 1 or 0);
            - sequential_indicating_sequences: the corresponding segments 
            where an indicating sequence contains consecutive values of 1;
            - global_sequential_indicating_sequence: the unified segments
            for all the input and output signals;
            - ks_indicating_sequences: the indicating sequences resulted
            from the lilliefors (Kolmogorov-Smirnov) test;
            - ts_indicating_sequences: the indicating sequences resulted
            from the t-student (two-mean comparison) test;
            - unified_indicating_sequence: the unified indicating sequence
            considering both ks_indicating_sequences and ts_indicating_sequences
            and considering all the input and output signals.
        """
        #Internal Variables
        self.data_segments_dict = defaultdict(dict)
        self.ks_values_dict = defaultdict(dict)
        self.ks_segments_indexes = defaultdict(dict)
        self.ts_values_dict = defaultdict(dict)
        self.sequential_indicating_sequences = defaultdict(dict)
        self.global_sequential_indicating_sequence = None
        self.ks_indicating_sequences = None
        self.ts_indicating_sequences = None
        self.unified_indicating_sequence = None
        
        
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
        else:
            Tc = None

        return Tc

    def _kolmogorov_smirnov_test(self, Tc):
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
        
        Output:
            lill_result: the Kolmogorov-Smirnov test result. If 1, the 
            Tc samples was able to reject the null hypothesis of 
            exponential distribution. If 0, the null hypothesis could not
            be rejected.
        """

        #Array for simulated Tc
        T_arr = range(0,60)

        #Estimated CDF
        f_hat = []

        #Interval Exponential Average
        lamb = 1/np.mean(Tc)

        #Theoretical CDF
        Ft = 1-np.exp(-lamb*T_arr)

        #Computed CDF
        for t in T_arr:
            a = []
            for tc in Tc:
                if tc <= t:
                    a.append(1)
                else:
                    a.append(0)
            f_hat.append(np.mean(a))

        #Lilliefors Computed Value
        Dt = np.max(np.abs(Ft-f_hat))
        
        #Compare with Critical Value
        if len(Tc) < 30:
            Dc = self.ks_critic/np.sqrt(len(Tc)-1)
        else:
            Dc = 2.5/np.sqrt(len(Tc)-1)

        if Dt < Dc:
            lill_result = 0
        else:
            lill_result = 1
        
        return lill_result
    
    def _magnitude_changes_test(self, data_segment):
        """
        Performs all the steps for checking if a data
        segment experiences a fair amount of magnitude
        changes.
        
        Arguments:
            data_segment: a data segment.
        
        Output:
            lill_result: the Kolmogorov-Smirnov test result. If 1, the 
            Tc samples was able to reject the null hypothesis of 
            exponential distribution. If 0, the null hypothesis could not
            be rejected.   
        """

        #Fid Mean Crossing Indexes
        mean_crossing_idx_arr = self._find_mean_crossing_indexes(data_segment=data_segment)
        
        #Create Mean Crossing Statistic
        Tc = self._create_mean_crossing_statistic(mean_crossing_idx_arr=mean_crossing_idx_arr)
        
        try: #If fails, Tc is Null
            if len(Tc) > 3:
                #Compute Kolmogorov-Smirnov (Lilliefors) Statistical Test
                lill_result = self._kolmogorov_smirnov_test(Tc=Tc)
            else:
                lill_result = 1
        except: #If Tc is Null, only one mean-crossing was found in the data
            lill_result = 1
            
        return lill_result
    
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
            
            #Perform Kolmogorov-Smirnov (Lilliefors) Test
            self.ks_values_dict[data_type][data_idx_name] = []
            for segment in self.data_segments_dict[data_type][data_idx_name]:
                self.ks_values_dict[data_type][data_idx_name].\
                    append(self._magnitude_changes_test(data[segment, col_idx]))
            
            #Take Segment Indexes that Satifies the Hypothesis Test
            self.ks_segments_indexes[data_type][data_idx_name] = \
                np.squeeze(np.argwhere(np.array(self.ks_values_dict[data_type][data_idx_name]) == 1))
            
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
        have a mean_delta difference in mean. The null hypothesis is that 
        the mean difference is lower or equal mean_delta. The indicating 
        sequence of each signal is updated case a particular segment meet 
        the test specifications.
        
        A t-student with unknown mean and variance is performed to compare
        the intervals and the resulting p-value is compared to the provided
        significance threshold (alpha).
        
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
        
        #Compute mean of intervals
        mean_1 = np.mean(interval_1)
        mean_2 = np.mean(interval_2)
        
        #Compute degrees of freedom
        w1 = (np.var(interval_1)**2)/len(interval_1)
        w2 = (np.var(interval_2)**2)/len(interval_2)
        
        df = ((w1+w2)**2)/(((w1**2)/(len(interval_1)+1))+\
                           ((w2**2)/(len(interval_2)+1))) - 2
        
        #Perform t-student test for unknown mean and variance
        tcalc = (np.abs(mean_1-mean_2)-self.mean_delta)/(np.sqrt(((np.var(interval_1)**2)/len(interval_1) + 
                                                        (np.var(interval_2)**2)/len(interval_2))))
        
        if tcalc > stats.t.ppf(1-self.alpha/2, df=df):
            data_signal_indexes = np.array(self.data_segments_dict[data_type][data_idx_name])
            true_data_indexes_1 = data_signal_indexes[segment_idx]
            true_data_indexes_2 = data_signal_indexes[segment_idx+1]
            self.indicating_sequences[data_type][data_idx_name][true_data_indexes_1] = 1.0
            self.indicating_sequences[data_type][data_idx_name][true_data_indexes_2] = 1.0
            return tcalc
        else:
            return tcalc
        
    
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
            
            self.ts_values_dict[data_type][data_idx_name] = list(executor(difference_in_mean_task))
        
    def _create_segments_dict(self, data, data_cols, data_type):
        """
        This function creates a data segment for each input and output
        signal based on the initial intervals provided.
        
        Arguments:
            data: a data matrix (either input or output data)
            data_cols: the columns names of the data matrix
            data_type: the data type (input or output)
        """
        for col in range(0,data.shape[1]):
            
            if data_cols is None:
                data_idx_name = data_type+"_"+str(col)
            else:
                data_idx_name = data_cols[col]
                
            self.data_segments_dict[data_type][data_idx_name] = self.initial_intervals[data_idx_name]  
    
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
        
        #Initialize Interval Variables
        self._initialize_internal_variables()
        
        #Verify data format
        X, y, X_cols, y_cols = self._verify_data(X,y)
        
        #Insert Noise
        if self.insert_noise_sp:
            X = self._insert_random_noise(data_segment=X,
                                        noise_mean=0,
                                        noise_std=self.noise_std)
        
        if self.insert_noise_y:
            y = self._insert_random_noise(data_segment=y,
                                        noise_mean=0,
                                        noise_std=self.noise_std)

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

#See below the used libraries Licenses
#-------------------------------------

#Scipy license
#-------------

# Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

#Pandas license
#--------------

# Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
# All rights reserved.
#
# Copyright (c) 2011-2020, Open source contributors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

#Numpy license
#-------------

# Copyright (c) 2005-2020, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# * Neither the name of the NumPy Developers nor the names of any
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

#Joblib license
#--------------

# Copyright (c) 2008-2016, The joblib developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.