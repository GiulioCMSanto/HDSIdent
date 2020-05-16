from HDSIdent.model_structures.model_structures import ModelStructure
 
from numpy.linalg import cond, qr, norm
from joblib import Parallel, delayed
from collections import defaultdict
from scipy import signal
import pandas as pd
import numpy as np
from sympy import *
import sympy

class LaguerreStructure(ModelStructure):
    """
    This class implements the Laguerre Filter Structure. The
    Laguerre Filter structure is defined as in the following
    references:
    
    PERETZKI, D. et al. Data mining of historic data for process identification.
    In: Proceedings of the 2011 AIChE Annual Meeting, p. 1027–1033, 2011.
    
    BITTENCOURT, A. C. et al. An algorithm for finding process identification intervals
    from normal operating data. Processes, v. 3, p. 357–383, 2015.
    
    WAHLBERG, B. System identification using laguerre models. IEEE Transactions on
    Automatic Control, IEEE, v. 36, n. 5, p. 551–562, 1991.

    Belonging to the ModelStructure class, the LaguerreStructure
    is able to compute the following metrics:
    
    1) Compute a Leguerre Regressor Matrix
    2) Compute Effective Rank of types 1 and 2
    3) Compute the Condition Number
    4) Compute a scalar correlation between each input and each output
    5) Estimate the regression parameters
    6) Compute a qui-squared test for the regression parameters
    
    Arguments:
        Nb: the Laguerre Filter order
        p: the Laguerre Filter pole (0 < p < 1)
        delay: the maximum/minimum cross-correlation lag value between the input and the output signals
        cc_alpha: the significance level to be considered for the cross-correlation test
        initial_intervals: the initial intervals indexes
        efr_type: the effective rank type (type_1 or type_2)
        sv_thr: singular value threshold for computing the effective rank
        Ts: the Laguerre Filter sampling period
        n_jobs: the number of CPUs to use
        verbose: the degree of verbosity (from 0 to 10)
    """
    def __init__(self,
                 Nb,
                 p,
                 delay,
                 cc_alpha,
                 initial_intervals,
                 efr_type,
                 sv_thr = 0.1,
                 Ts = 1,
                 n_jobs = -1,
                 verbose = 0):

        self.Nb = Nb
        self.p = p
        self.delay = delay
        self.cc_alpha = cc_alpha
        self.initial_intervals = initial_intervals
        self.efr_type = efr_type
        self.sv_thr = sv_thr
        self.Ts = Ts
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.name = 'Laguerre'
        
        #Define shift operator
        #q*x[k] = x[k+1]
        self.q = symbols('q')
                
    def _laguerre_filter_tf(self, order):
        """
        Implements a discrete-time Laguerre Filter transfer function.
        
        Arguments:
            order: the Laguerre Filter order
            
        Output:
            L_ts: the laguerre filter transfer function (scipy TransferFunctionDiscrete)
        """     
        
        #Filter Constant
        constant = np.sqrt((1-self.p**2)*self.Ts)
        
        #Filter Numerator
        num = poly(constant*expand((1-self.p*self.q)**(order-1)),self.q)
        
        #Filter Denominator
        den = poly(expand((self.q-self.p)*((self.q-self.p)**(order-1))),self.q)
        
        #Filter Tranfer Function
        num = np.array(num.coeffs(), dtype=float)
        den = np.array(den.coeffs(), dtype=float)    
        L_tf = signal.TransferFunction(num,den,dt=self.Ts)
        
        return L_tf
    
    def _compute_leguerre_regressor_matrix(self, X,y,input_idx,X_cols,output_idx,y_cols,segment):
        """
        Computes the Laguerre Regressor Matrix for a given
        signal segment X.
        
        Arguments:
            X: a matrix of input signals. Each signal is a column;
            input_idx: the sequential number of the execution input;
            X_cols: the input data columns in case they are provided;
            output_idx: the sequential number of the execution output;
            y_cols: the output data columns in case they are provided;
            segment: the sequential number of the execution segment (interval).
        
        Output:
            Phi: the corresponding regressor matrix for the given segment of signal.
        """
        
        #Take Column Names
        input_idx_name, output_idx_name = self._update_index_name(input_idx, X_cols, output_idx, y_cols)
        
        #Take X and y signal segments
        X_seg = X[:,input_idx][self.initial_intervals[segment]]
        
        #Initialize Regressor Matrix
        Phi = np.zeros((len(X_seg)-1, self.Nb))
        
        for order in range(1,self.Nb+1):
            #Compute the Laguerre Filter Transfer Function
            L_tf = self._laguerre_filter_tf(order=order)

            #Simulate Laguerre Filter for Signal of Column col 
            _, X_out = signal.dlsim(system=L_tf,
                                    u=X_seg,
                                    t=range(len(X_seg)))

            Phi[:,order-1] = np.squeeze(X_out[1:])   
        
        #Update interval variable
        self.Phi_dict['segment'+'_'+str(segment)] \
                     [output_idx_name] \
                     [input_idx_name] = Phi
        
    def _compute_Laguerre_miso_ranks(self,X,y,input_idx,X_cols,output_idx,y_cols,segment):
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
        Laguerre_regressor_mtx = self.Phi_dict['segment'+'_'+str(segment)] \
                                              [output_idx_name] \
                                              [input_idx_name]
        
        #Compute Ranks
        self._compute_miso_ranks(X = X,
                                 y = y,
                                 regressor_mtrx = Laguerre_regressor_mtx,
                                 input_idx = input_idx,
                                 X_cols = X_cols,
                                 output_idx = output_idx,
                                 y_cols = y_cols,
                                 segment = segment)
    
    def fit(self, X, y):
        """
        This function performs the following rotines:
            - Computes the Laguerre Regressor Matrix for the given data;
            - Computes the effective rank for the regressor matrix;
            - Computes the cross-correlation scalar metric for each input and output data;
            - Computes the Condition Number for each Regressor Matrix from each segment;
            - Computes the Laguerre parameters estimations;
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
            print("Computing Laguerre Regressor Matrix...")
            
        #Compute Laguerre Regressor Matrix
        lg_regressor_task = (delayed(self._compute_leguerre_regressor_matrix)(X, y, input_idx,
                                                                              X_cols, output_idx,
                                                                              y_cols, segment)
                              for segment in self.initial_intervals.keys()
                              for input_idx in range(0,X.shape[1])
                              for output_idx in range(0,y.shape[1]))
        executor(lg_regressor_task)
        
        if self.verbose > 0:
            print("Performing QR-Decomposition...")
            
        #Make QR_Factorization (Condition Number and Qui-squared Test)
        cond_numb_task = (delayed(self._qr_factorization)(y, input_idx,
                                                          X_cols, output_idx,
                                                          y_cols, segment)
                          for segment in self.initial_intervals.keys()
                          for input_idx in range(0,X.shape[1])
                          for output_idx in range(0,y.shape[1]))
        executor(cond_numb_task)   
        
        if self.verbose > 0:
            print("Computing Effective Rank...")
            
        #Compute the Effective Rank for each MISO system 
        miso_ranks_task = (delayed(self._compute_Laguerre_miso_ranks)(X, y, input_idx, 
                                                                      X_cols, output_idx,
                                                                      y_cols, segment)
                           for segment in self.initial_intervals.keys()
                           for input_idx in range(0,X.shape[1])
                           for output_idx in range(0,y.shape[1]))
        
        executor(miso_ranks_task)    
        
        if self.verbose > 0:
            print("Computing Cross-Correlation Metric...")
            
        #Compute cross-correlation scalar metric for each MISO system
        miso_corr_task = (delayed(self._compute_miso_correlations)(X, y, input_idx,
                                                                   X_cols, output_idx,
                                                                   y_cols, segment)
                          for segment in self.initial_intervals.keys()
                          for input_idx in range(0,X.shape[1])
                          for output_idx in range(0,y.shape[1]))
        executor(miso_corr_task)
        
        if self.verbose > 0:
            print("Laguerre fit finished!")
            
        return (self.miso_ranks, self.miso_correlations, self.cond_num_dict, self.qui_squared_dict)