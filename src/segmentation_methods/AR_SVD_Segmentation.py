from joblib import Parallel, delayed
from collections import defaultdict
from scipy.stats import norm
import pandas as pd
import numpy as np
import scipy

class AR_SVD_segmentation(object):
    """
    This class receives intervals of data and verifies if these intervals
    are suitable for performing System Identification. The Autoregressive (AR)
    structure is used to compute the regressor matrix. The effective rank of the
    AR information matrix is computed and compared to a user-defined threshold.
    Finally, correlation between the input and the output is computed using a scalar
    metric and it is also compared to a user-defined threshold.
    
    The method proposed in this class was published in:

    RIBEIRO, A. H.; AGUIRRE, L. A. Selecting transients automatically
    for the identification of models for an oil well. IFAC-PapersOnLine,
    v. 48, n. 6, p. 154–158, 2015.
    """
    def __init__(self,
                 ny,
                 delay,
                 alpha,
                 initial_intervals,
                 efr_type,
                 efr_thr,
                 cc_thr,
                 sv_thr = 0.1,
                 normalize=True,
                 n_jobs = -1,
                 verbose = 0):
        """
        Constructor.
        
        Arguments:
            ny: the Autoregressive structure order
            delay: the maximum/minimum cross-correlation lag value between the input and the output signals
            alpha: the significance level to be considered for the cross-correlation test
            initial_intervals: the initial interval indexes
            efr_type: the effective rank type (type_1 or type_2)
            efr_thr: the effective rank threshold required to accept an interval for system identification
            cc_thr: cross-correlation threshold required to accept an interval for system identification
            sv_thr: singular value threshold for computing the effective rank
            n_jobs: the number of CPUs to use
            verbose: the degree of verbosity (from 0 to 10)
        """
        
        self.ny = ny
        self.delay = delay
        self.alpha = alpha
        self.initial_intervals = initial_intervals
        self.efr_type = efr_type
        self.efr_thr = efr_thr
        self.cc_thr = cc_thr
        self.sv_thr = sv_thr
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self._miso_ranks = defaultdict(dict)
        self._miso_correlations = defaultdict(lambda: defaultdict(dict))
        self._segments_indexes = defaultdict()
        self._segments = defaultdict(dict)
        
    def _AR_regressor_matrix(self, y, ny, X=None):
        """
        Computes the autoregressive regressor matrix.
        Notice that the autoregressive structure do not depends
        on the input signal.

        The autoregressive vector for an instant k can be calculated
        as follows:

        Phi(k) = [y(k-1), ..., y(k-ny)]
        
        Arguments:
            y: the output signal
            ny: the regressor order

        Output:
            ar_matrix: the autoregressive regressor matrix
        """
        ar_matrix = np.zeros((len(y)-ny,ny))
        for idx in range(ny,len(y)):
            ar_matrix[idx-ny,:] = y[idx-ny:idx][::-1].reshape(1,-1)

        return ar_matrix
    
    def _information_matrix(self, Phi):
        """
        Computes the Information Matrix for a given regressor
        matrix Phi. The information matrix is defined as [(Phi)^T][Phi]
        
        Arguments:
            Phi: the given regressor matrix
            
        Output:
            I: the information matrix for the given regressor
        """
        
        I = np.matmul(Phi.T,Phi)
        
        return I
    
    def _cross_correlation_scalar_metric(self, X, y, delay, alpha):
        """
        Computes a scalar metric that represents the cross-correlation
        function for a range of lag values. The lag range goes from
        -delay to delay. The normalized cross-correlation is computed
        for signals X and y and compared to the critical value of a
        two-sided normal distribution for an alpha confidence value.

        This metric is proposed in the following reference:

        RIBEIRO, A. H.; AGUIRRE, L. A. Selecting transients automatically
        for the identification of models for an oil well. IFAC-PapersOnLine,
        v. 48, n. 6, p. 154–158, 2015.

        Arguments:
            X: the input signal
            y: the output signal
            delay: the maximum/minimum cross-correlation lag value between the input and the output signals
            alpha: the confidence value for a normal distribution

        Output:
            ccsm: the cross-correlation scalar metric
        """
        #Compute p-value
        p = norm.ppf(1-(1-alpha)/2)/np.sqrt(len(X))

        s_arr = []
        for d in range(-delay,delay+1):
            #Compute Normalized Cross Corellation for current delay
            ncc = self._normalized_cross_correlation(X=X, y=y, delay=d)

            if np.abs(ncc) <= p:
                s_arr.append(0)
            elif np.abs(ncc) > p and d != 0:
                s_arr.append(np.abs(ncc - p)/np.abs(d))
            else:
                s_arr.append(np.abs(ncc - p))

        ccsm = np.sum(s_arr)

        return ccsm
    
    def _normalized_cross_correlation(self, X, y, delay):
        """
        Computes the normalized cross-correlation function
        of signals X and y for a given delay value.

        Arguments:
            X: the input signal
            y: the output signal
            delay: the delay between both signals

        Output:
            ncc: the normalized cross-correlation value
        """

        if delay < 0:
            num = np.sum([X[idx]*y[idx+delay] for idx in range(delay,len(X))])
            den_1 = np.sum([X[idx]**2 for idx in range(delay,len(X))])
            den_2 = np.sum([y[idx+delay]**2 for idx in range(delay,len(X))])
            den = np.sqrt(den_1*den_2)
        else:
            num = np.sum([X[idx]*y[idx+delay] for idx in range(0,len(X)-delay)])
            den_1 = np.sum([X[idx]**2 for idx in range(0,len(X)-delay)])
            den_2 = np.sum([y[idx+delay]**2 for idx in range(0,len(X)-delay)])
            den = np.sqrt(den_1*den_2)
        
        if den == 0:
            ncc = 0
        else:
            ncc = num/den

        return ncc

    def _effective_rank_type_2(self, singular_values, threshold):
        """
        Compute the effective rank as a function of the difference
        of two consecutive singular values.

        This implementation was based on the following reference:

        RIBEIRO, A. H.; AGUIRRE, L. A. Selecting transients automatically
        for the identification of models for an oil well. IFAC-PapersOnLine,
        v. 48, n. 6, p. 154–158, 2015.

        Arguments:
            singular_values: matrix si
            ngular values
            threshold: effective rank threshold

        Output:
            efr: the computed effective rank
        """

        efr_arr = []
        for idx in range(1,len(singular_values)):
            #Compute Consecutives Singular Values
            s_i_1 = singular_values[idx-1]
            s_i = singular_values[idx]

            #Compute the difference of the consecutive singular values
            s_diff = s_i_1 - s_i

            #Compute effective rank for index idx
            if s_diff > threshold:
                efr_arr.append(1)
            else:
                efr_arr.append(0)

        efr = np.sum(efr_arr)

        return efr

    def _effective_rank_type_1(self, singular_values, threshold):
        """
        Compute the effective rank as a function of the normalized
        singular values.

        This implementation was based on the following reference:

        RIBEIRO, A. H.; AGUIRRE, L. A. Selecting transients automatically
        for the identification of models for an oil well. IFAC-PapersOnLine,
        v. 48, n. 6, p. 154–158, 2015.

        Arguments:
            singular_values: matrix singular values
            threshold: effective rank threshold

        Output:
            efr: the computed effective rank
        """

        #Compute L1-norm
        l1_norm = np.sum([np.abs(s) for s in singular_values])

        #Compute Normalized Singular Values
        p_arr = [s/l1_norm for s in singular_values]

        #Compute Effective Rank for given Threshold
        efr = np.sum([1 if p > threshold else 0 for p in p_arr])

        return efr

    def _effective_rank(self, A, threshold, efr_type):
        """
        Compute the effective rank of matrix A for
        a given threshold. Two types of effective 
        rank are available and implemented based on the
        following reference:

        RIBEIRO, A. H.; AGUIRRE, L. A. Selecting transients automatically
        for the identification of models for an oil well. IFAC-PapersOnLine,
        v. 48, n. 6, p. 154–158, 2015.

        Arguments:
            A: the input matrix
            threshold: the threshold for computing the effective rank
            efr_type: type_1 or type_2

        Output:
            efr: the effective rank
        """

        #Compute Singular Values of Matrix A
        _, singular_values, _ = linalg.svd(A)

        #Compute Effective Rank
        if efr_type == 'type_1':
            return self._effective_rank_type_1(singular_values = singular_values,
                                               threshold = threshold)
        elif efr_type == 'type_2':
            return self._effective_rank_type_2(singular_values = singular_values,
                                               threshold = threshold)
    
    def _compute_miso_ranks(self, y, output_idx, segment):
        """
        For each MISO System, i.e., for each output, compute the effective rank
        of the AR Information matrix for the corresponding output.
        
        Arguments:
            y: the output signal
            output_idx: the sequential number of the execution output
            segment: the sequential number of the execution segment (interval)
        """
        
        #Compute AR Regressor Matrix for the running interval
        AR_regressor_mtx = self._AR_regressor_matrix(y = y[:,output_idx] \
                                                          [self.initial_intervals[segment]],
                                                     ny = self.ny)
        
        #Compute the corresponding Information Matrix
        I_AR_mtx = self._information_matrix(AR_regressor_mtx)
        
        #Compute the Effective Rank of the Information Matrix
        efr = self._effective_rank(A = I_AR_mtx,
                                   threshold = self.sv_thr,
                                   efr_type = self.efr_type)
        
        self._miso_ranks['segment'+'_'+str(segment)]['output'+'_'+str(output_idx)] = efr
    
    def _compute_miso_correlations(self, X, y, input_idx, output_idx, segment):
        """
        For each MISO System, i.e., for each output, compute the cross-correlation
        metric between each input and the corresponding output.
        
        Arguments:
            X: the input signal
            y: the output signal
            input_idx: the sequential number of the execution input
            output_idx: the sequential number of the execution output
            segment: the sequential number of the execution segment (interval)
        """
        ncc = self._cross_correlation_scalar_metric(X = X[:,input_idx][self.initial_intervals[segment]],
                                                    y = y[:,output_idx][self.initial_intervals[segment]],
                                                    delay = self.delay,
                                                    alpha = self.alpha)
        
        self._miso_correlations['segment'+'_'+str(segment)] \
                               ['output'+'_'+str(output_idx)] \
                               ['input'+'_'+str(input_idx)] = ncc
    
    def _compute_segment_indexes(self, segment, input_idx, output_idx):
        """
        Compares the Effective Ranks and cross-correlation metrics with
        the provided thresholds.
        
        Arguments:
            segment: the sequential number of the execution segment (interval)
            input_idx: the sequential number of the execution input
            output_idx: the sequential number of the execution output
        """
        if ((teste._miso_ranks['segment'+'_'+str(segment)] \
                              ['output'+'_'+str(output_idx)] >= self.efr_thr)
            and \
            (teste._miso_correlations['segment'+'_'+str(segment)] \
                                     ['output'+'_'+str(output_idx)] \
                                     ['input'+'_'+str(input_idx)] >= self.cc_thr)):
                
            self._segments_indexes['segment'+'_'+str(segment)] = self.initial_intervals[segment]
    
    def _compute_segments(self, X, y, X_cols, y_cols):
        """
        Compute a dictionary with the Input/Output signals for the obtained intervals.
        
        Arguments:
            X: the input data in numpy array format
            y: the input data in numpy array format
            X_cols: the input data columns in case they are provided
            y_cols: the output data columns in case they are provided
        """
        for segment in self._segments_indexes.keys():
            
            for output_idx in range(0,y.shape[1]):
                if y_cols is not None:
                    output_idx_name = y_cols[output_idx]
                else:
                    output_idx_name = 'output'+'_'+str(output_idx)
                    
                self._segments[segment][output_idx_name] = y[:,output_idx][self._segments_indexes[segment]]
                
            for input_idx in range(0,X.shape[1]):
                if X_cols is not None:
                    input_idx_name = X_cols[input_idx]
                else:
                    input_idx_name = 'input'+'_'+str(input_idx)
                    
                self._segments[segment][input_idx_name] = X[:,input_idx][self._segments_indexes[segment]]
    
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
            X = X.values.reshape(-1,1)
        elif type(X) == np.ndarray:
            X = X.reshape(-1,1)
            X_cols = None
        else:
            raise Exception("Input data must be a pandas dataframe or a numpy array") 

        if type(y) == pd.core.frame.DataFrame:
            y_cols = y.columns
            y = y.values.reshape(-1,1)
        elif type(y) == np.ndarray:
            y = y.reshape(-1,1)
            y_cols = None
        else:
            raise Exception("Input data must be a pandas dataframe or a numpy array") 
            
        return X, y, X_cols, y_cols
    
    def fit(self, X, y):
        """
        This function performs the following rotines:
            - Compute the AR Regressor Matrix for the given data
            - Computes the effective rank for the regressor matrix
            - Computes the cross-correlation scalar metric for the input and output data
            - Compare the effective rank and cross-correlation with the provided thresholds
        
        Arguments:
            X: the input signal
            y: the output signal
        
        Output:
            _segments: the input/output signals for the corresponding segments obtained by the algorithm
        """
        
        #Reset metrics
        self._miso_ranks = defaultdict(dict)
        self._miso_correlations = defaultdict(lambda: defaultdict(dict))
        self._segments_indexes = defaultdict()
        self._segments = defaultdict(dict)
        
        #Verify data format
        X, y, X_cols, y_cols = self._verify_data(X,y)
        
        #Make an Parallel executor
        executor = Parallel(require='sharedmem',
                            n_jobs=self.n_jobs,
                            verbose=self.verbose)
        
        #Compute the Effective Rank for each MISO system 
        miso_ranks_task = (delayed(self._compute_miso_ranks)(y,output_idx,segment)
                           for segment in range(0,len(self.initial_intervals))
                           for output_idx in range(0,y.shape[1]))
        executor(miso_ranks_task)
        
        #Compute cross-correlation scalar metric for each MISO system
        miso_corr_task = (delayed(self._compute_miso_correlations)(X,y,input_idx,output_idx,segment)
                          for segment in range(0,len(self.initial_intervals))
                          for output_idx in range(0,y.shape[1])
                          for input_idx in range(0,X.shape[1]))
        executor(miso_corr_task)
        
        #Compare Effective Ranks and cross-correlation metrics with thresholds
        segmentation_task = (delayed(self._compute_segment_indexes)(segment,input_idx,output_idx)
                             for segment in range(0,len(self.initial_intervals))
                             for output_idx in range(0,y.shape[1])
                             for input_idx in range(0,X.shape[1]))
        executor(segmentation_task)
        
        #Compute dictionary with Input and Output signals for the obtained intervals
        self._compute_segments(X=X, y=y, X_cols=X_cols, y_cols=y_cols)
        
        return self._segments