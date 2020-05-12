from HDSIdent.model_structures.model_structures import ModelStructure

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

class ARXStructure(ModelStructure):
    """
    This class implements the Autoregressive With External Input (ARX) Structure. 

    Belonging to the ModelStructure class, the ARXStructure
    is able to compute the following metrics:
    
    1) Compute a ARX Regressor Matrix
    2) Compute Effective Rank of types 1 and 2
    3) Compute the Condition Number
    4) Compute a scalar correlation between each input and each output
    5) Estimate the regression parameters
    6) Compute a qui-squared test for the regression parameters
    
    Arguments:
        nx: the ARX input order
        ny: the ARX output order
        delay: the maximum/minimum cross-correlation lag value between the input and the output signals
        cc_alpha: the significance level to be considered for the cross-correlation test
        initial_intervals: the initial intervals indexes
        efr_type: the effective rank type (type_1 or type_2)
        sv_thr: singular value threshold for computing the effective rank
        n_jobs: the number of CPUs to use
        verbose: the degree of verbosity (from 0 to 10)
    """
    def __init__(self,
                 nx,
                 ny,
                 delay,
                 cc_alpha,
                 initial_intervals,
                 efr_type,
                 sv_thr = 0.1,
                 n_jobs = -1,
                 verbose = 0):

        self.nx = nx
        self.ny = ny
        self.Nb = ny+nx
        self.delay = delay
        self.cc_alpha = cc_alpha
        self.initial_intervals = initial_intervals
        self.efr_type = efr_type
        self.sv_thr = sv_thr
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.name = 'ARX'
                
    def _ARX_regressor_matrix(self, X, y,input_idx,X_cols,output_idx,y_cols,segment):
        """
        Computes the Autoregressive With External Input
        (ARX) structure regressor matrix. Notice that the
        The ARX vector for an instant k can be calculated
        as follows:
        phi(k) = [y(k-1), ..., y(k-ny), X(k-1), ..., X(k-nx)]
        
        Arguments:
            X: a matrix of input signals. Each signal is a column;
            y: a matrix of output signals. Each signal is a column;
            input_idx: the sequential number of the execution input;
            X_cols: the input data columns in case they are provided;
            output_idx: the sequential number of the execution output;
            y_cols: the output data columns in case they are provided;
            segment: the sequential number of the execution segment (interval).
        """
        #Take Column Names
        input_idx_name, output_idx_name = self._update_index_name(input_idx, X_cols, output_idx, y_cols)
        
        #Take X and y signal segments
        X_seg = X[:,input_idx][self.initial_intervals[segment]]
        y_seg = y[:,output_idx][self.initial_intervals[segment]]
        
        #Compute Regressor Matrix for the given segment
        arx_matrix = np.zeros((len(y_seg)-max(self.nx,self.ny),self.ny+self.nx))
        for idx in range(max(self.nx,self.ny),len(y_seg)):
            #Create y component
            arx_matrix[idx-max(self.nx,self.ny),:self.ny] = y_seg[idx-self.ny:idx][::-1].reshape(1,-1)
            #Create X component
            arx_matrix[idx-max(self.nx,self.ny),self.ny:] = X_seg[idx-self.nx:idx][::-1].reshape(1,-1)

        #Update interval variable
        self.Phi_dict['segment'+'_'+str(segment)] \
                     [output_idx_name] \
                     [input_idx_name] = arx_matrix
    
    def _compute_ARX_miso_ranks(self,X,y,input_idx,X_cols,output_idx,y_cols,segment):
        """
        This function computes the effective rank
        for each (input/output) regressor matrix.
        
        Arguments:
            X: a matrix of input signals. Each signal is a column;
            y: a matrix of output signals. Each signal is a column;
            input_idx: the sequential number of the execution input;
            X_cols: the input data columns in case they are provided;
            output_idx: the sequential number of the execution output;
            y_cols: the output data columns in case they are provided;
            segment: the sequential number of the execution segment (interval).
        """
        #Take Column Names
        input_idx_name, output_idx_name = self._update_index_name(input_idx, X_cols, output_idx, y_cols)
        
        #Take Regressor Matrix
        ARX_regressor_mtx = self.Phi_dict['segment'+'_'+str(segment)] \
                                         [output_idx_name] \
                                         [input_idx_name]
        
        #Compute Ranks
        self._compute_miso_ranks(X = X,
                                 y = y,
                                 regressor_mtrx = ARX_regressor_mtx,
                                 input_idx = input_idx,
                                 X_cols = X_cols,
                                 output_idx = output_idx,
                                 y_cols = y_cols,
                                 segment = segment)
    
    def fit(self, X, y):
        """
        This function performs the following rotines:
            - Computes the ARX Regressor Matrix for the given data;
            - Computes the effective rank for the regressor matrix;
            - Computes the cross-correlation scalar metric for each input and output data;
            - Computes the Condition Number for each Regressor Matrix from each segment;
            - Computes the ARX parameters estimations;
            - Computes the qui-squared test for validating the estimated parameters;
        
        Arguments:
            X: the input signal
            y: the output signal
        
        Output:
            (self.miso_ranks: the effective rank for each (input/output) regressor,
             self.miso_correlations: the scalar metric cross-correlation for each input/output, 
             self.cond_num_dict: the Condition Number for each (input/output) regressor, 
             self.qui_squared_dict: the qui-squared test for validating the estimated parameters)
        """
        #Verify data format
        X, y, X_cols, y_cols = self._verify_data(X,y)
        
        #Initialize Internal Variables
        self._initialize_metrics(X,y,X_cols,y_cols)

        #Make an Parallel executor
        executor = Parallel(require='sharedmem',
                            n_jobs=self.n_jobs,
                            verbose=self.verbose)
        
        if self.verbose > 0:
            print("Computing ARX Regressor Matrix...")
            
        #Compute ARX Regressor Matrix
        arx_regressor_task = (delayed(self._ARX_regressor_matrix)(X,y,input_idx,X_cols,output_idx,y_cols,segment)
                              for segment in self.initial_intervals.keys()
                              for input_idx in range(0,X.shape[1])
                              for output_idx in range(0,y.shape[1]))
        executor(arx_regressor_task)
        
        if self.verbose > 0:
            print("Performing QR-Decomposition...")
            
        #Make QR_Factorization
        cond_numb_task = (delayed(self._qr_factorization)(y,input_idx,X_cols,output_idx,y_cols,segment)
                          for segment in self.initial_intervals.keys()
                          for input_idx in range(0,X.shape[1])
                          for output_idx in range(0,y.shape[1]))
        executor(cond_numb_task)    
        
        if self.verbose > 0:
            print("Computing Effective Rank...")
            
        #Compute the Effective Rank for each MISO system 
        miso_ranks_task = (delayed(self._compute_ARX_miso_ranks)(X,y,input_idx,X_cols,output_idx,y_cols,segment)
                           for segment in self.initial_intervals.keys()
                           for input_idx in range(0,X.shape[1])
                           for output_idx in range(0,y.shape[1]))
        
        executor(miso_ranks_task)    
        
        if self.verbose > 0:
            print("Computing Cross-Correlation Metric...")
            
        #Compute cross-correlation scalar metric for each MISO system
        miso_corr_task = (delayed(self._compute_miso_correlations)(X,y,input_idx,X_cols,output_idx,y_cols,segment)
                          for segment in self.initial_intervals.keys()
                          for input_idx in range(0,X.shape[1])
                          for output_idx in range(0,y.shape[1]))
        executor(miso_corr_task)
        
        if self.verbose > 0:
            print("ARX fit finished!")
            
        return (self.miso_ranks, self.miso_correlations, self.cond_num_dict, self.qui_squared_dict)