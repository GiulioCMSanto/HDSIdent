from numpy.linalg import cond, qr, norm
from joblib import Parallel, delayed
from collections import defaultdict
from numpy import linalg as LA
from scipy.stats import chi2
from scipy import signal
import pandas as pd
import numpy as np
from sympy import *
import sympy

class LaguerreFilter(object):
    """
    Implements a discrete-time Laguerre Filter transfer function.
    
    Arguments:
        p: the laguerre filter pole (0 < p < 1)
        Nb: the laguerre filter order
        Ts: the sampling period
        verbose: the degree of verbosity
    
    Output:
        L: the laguerre filter transfer function (scipy TransferFunctionDiscrete)
    """
    
    def __init__(self, p, Nb, Ts, verbose = 0, n_jobs = -1):
        
        self.q = symbols('q')
        self.p = p
        self.Nb = Nb
        self.Ts = Ts
        self.verbose = verbose
        self.n_jobs = n_jobs
        
    def _laguerre_filter_tf(self, order):
        """
        Implements a discrete-time Laguerre Filter transfer function.

        Arguments:
            p: the laguerre filter pole (0 < p < 1)
            Nb: the laguerre filter order
            Ts: the sampling period
            
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
    
    def _compute_regressor_matrix(self, X, Phi_dict, input_idx, X_cols, order):
        """
        Computes the Regressor Matrix for a Given matrix of
        signals X, for a given column (col), which corresponds
        to a particular signal in that matrix, and for a given
        structure (model) order.
        
        Arguments:
            X: a matrix of signals. Each signal is a column
            Phi_dict: an input dictionary for storing the regressor matrix
            input_idx: a particular input signal index
            order: the model (Laguerre Structure) order
        
        Output:
            Phi_dict: a dictionary with the columns indexes as keys
            and the corresponding regressor matrix for that signal
            as values.
        """
        #Compute the Laguerre Filter Transfer Function
        L_tf = self._laguerre_filter_tf(order=order)
        
        #Simulate Laguerre Filter for Signal of Column col 
        _, X_out = signal.dlsim(system=L_tf,
                                u=X[:,input_idx],
                                t=range(len(X[:,input_idx])))
        
        #Update the Dictionary of Regressor Matrices for Signal of Column col
        if X_cols is not None:
            input_name = X_cols[input_idx]
        else:
            input_name = input_idx
            
        Phi_dict[input_name][:,order-1] = np.squeeze(X_out[1:])      
    
    def _qr_factorization(self, y, input_idx, output_idx, X_cols, y_cols):
        """
        Performs a QR-Factorization (Decomposition) using numpy linear
        algebra library and uses the R matrix to solve the Ordinary Least
        Square (OLS) problem.
        
        Arguments:
            y: the ouput signals
            input_idx: the input signal index
            output_idx: the output signal index
        """
        
        if X_cols is not None:
            input_name = X_cols[input_idx]
        else:
            input_name = input_idx
        
        if y_cols is not None:
            output_name = y_cols[output_idx]
        else:
            output_name = output_idx
        
        #Create the Augmented Regressor Matrix [Phi y]
        self.Phi_aug_dict[input_name][output_name][:,:self.Nb] = self.Phi_dict[input_name]
        self.Phi_aug_dict[input_name][output_name][:,-1] = np.squeeze(y[:,output_idx][1:])
        
        #QR-Factorization
        Q, R = LA.qr(self.Phi_aug_dict[input_name][output_name])
        R1 = R[:self.Nb,:self.Nb]
        R2 = R[:self.Nb,self.Nb]
        R3 = R[self.Nb,self.Nb]
        
        #Comput Theta, Information Matrix and its Condition Number and the Qui-squared Test
        self.I_dict[input_name][output_name] = (1/len(np.squeeze(y[:,output_idx][1:])))*np.matmul(R1.T,R1)
        self.cond_num_dict[input_name][output_name] = LA.cond(self.I_dict[input_name][output_name])
        
        try:
            self.theta_dict[input_name][output_name] = np.matmul(LA.inv(R1),R2)
        except:
            pass
        
        self.qui_squared_dict[input_name][output_name] = (np.sqrt(len(np.squeeze(y[:,output_idx][1:])))/
                                                          np.abs(R3))*LA.norm(x=R2, ord=2)
        

    def _initialize_metrics(self, X, y, X_cols, y_cols):
        """
        This function initializes the following metrics:
            - Phi_dict: a dictionary of regressor matrices.
            - Phi_aug_dict: a dictionary of augmented matrices of the form [Phi y].
            - I_dict: a dictionary of information matrices of the form [Phi]^T[Phi].
            - cond_num_dict: a dictionary of condition numbers for each information matrix.
            - theta_dict: a dictionary of estimated parameter vectors phi = [ph1 ph2 ... phiNb].
        
        Output:
            Phi_dict: a dictionary of regressor matrices where each key corresponds to an input signal.
            Phi_aug_dict: a double dictionary. The first keys correspond to each input signal and the
            second keys correspond to each output signal.
            I_dict: a double dictionary of information matrices. The first keys correspond to each input
            signal and the second keys correspond to each output signal.
            cond_num_dict: a double dictionary of condition numbers.
            theta_dict: a double dictionary of estimated parameters.
        """
        Phi_dict = defaultdict()
        Phi_aug_dict = defaultdict(dict)
        I_dict = defaultdict(dict)
        cond_num_dict = defaultdict(dict)
        theta_dict = defaultdict(dict)
        qui_squared_dict = defaultdict(dict)
        
        for input_idx in range(X.shape[1]):
            
            if X_cols is not None:
                input_name = X_cols[input_idx]
            else:
                input_name = input_idx
                
            Phi_dict[input_name] = np.zeros((X.shape[0]-1, self.Nb))
            
            for output_idx in range(y.shape[1]):
                
                if y_cols is not None:
                    output_name = y_cols[output_idx]
                else:
                    output_name = output_idx
                
                Phi_aug_dict[input_name][output_name] = np.zeros((X.shape[0]-1, self.Nb + 1))
                I_dict[input_name][output_name] = np.zeros((self.Nb, self.Nb))
                cond_num_dict[input_name][output_name] = None
                theta_dict[input_name][output_name] = np.zeros((1,self.Nb))
                qui_squared_dict[input_name][output_name] = None
                
        return Phi_dict, Phi_aug_dict, I_dict, cond_num_dict, theta_dict, qui_squared_dict

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
    
    def fit(self,X,y):
        """
        Takes a time-series input signal (X) and
        computes a Nb order regressor matrix.
        
        Arguments:
            X: a matrix of input signals. Each signal corresponds to a column.
        
        Output:
            Phi_dict: a dictionary with the columns indexes as keys
            and the corresponding regressor matrix for that signal
            as values.            
        """
        
        #Verify data format
        X, y, X_cols, y_cols = self._verify_data(X,y)

        #Initialize Metrics
        (self.Phi_dict,
         self.Phi_aug_dict,
         self.I_dict,
         self.cond_num_dict,
         self.theta_dict,
         self.qui_squared_dict) = self._initialize_metrics(X,y,X_cols,y_cols)
        
        #Make an Parallel executor
        executor = Parallel(require='sharedmem',
                            n_jobs=self.n_jobs,
                            verbose=self.verbose)
        
        #Compute the Regressor Matrix for Each Input Signal in X
        laguerre_regressor_mtrx = (delayed(self._compute_regressor_matrix)(X,self.Phi_dict,input_idx,X_cols,order)
                                   for input_idx in range(X.shape[1])
                                   for order in range(1,self.Nb+1))
        
        executor(laguerre_regressor_mtrx)
        
        #Compute Metrics
        executor = Parallel(require='sharedmem',
                            n_jobs=self.n_jobs,
                            verbose=self.verbose)
        
        #Compute the Regressor Matrix for Each Input Signal in X
        metrics = (delayed(self._qr_factorization)(y,input_idx,output_idx,X_cols,y_cols)
                   for input_idx in range(X.shape[1])
                   for output_idx in range(y.shape[1]))        
        
        executor(metrics)
        
        return self.cond_num_dict, self.qui_squared_dict