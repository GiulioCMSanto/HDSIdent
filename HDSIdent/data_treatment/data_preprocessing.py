from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from collections import defaultdict
from scipy import signal
import pandas as pd
import numpy as np

class Preprocessing(object):
    """
    Performs preprocessing transformations in a given dataframe.
    The following transformations are applied:

        - An array of bad data (data with missing values) is created;
        - Data Scaling (Either StandardScaler or MinMaxScaler);
        - Removal of first samples to avoid deflection;
        - Butterworth low-pass filtering.
    
    Arguments:
        scaler: the scaler type to be used
        feature_range: the feature range as a tuple (min, max)
        k: the number of initial samples to be removed
        W: input frequency
        N: Butterworth filter order
        Ts: the sampling frequency of the digital filter (Default = 1.0 seconds)
    """
    def __init__(self, scaler='StandardScaler', feature_range = (0,1), k=10, W=None, N=1, Ts=1):
            
        self.scaler = scaler
        self.feature_range = feature_range
        self.k = k
        self.W = W
        self.N = N
        self.Ts = Ts
        
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
            
    def _scale(self, data):
        """
        Scales the dataframe according to the scaler type
        provided.
        """
        if self.scaler == 'MinMaxScaler':
            scl = MinMaxScaler(feature_range = self.feature_range)
            data = scl.fit_transform(data)
        elif self.scaler == 'StandardScaler':
            scl = StandardScaler()
            data = scl.fit_transform(data)
        else:
            raise Exception("Only MinMaxScaler and StandardScaler are accepted") 
        
        return data
    
    def _remove_first_samples(self, data):
        """
        Removes the initial samples to avoid deflection.
        """
        return data[self.k:,:]
        
    def _lowpass_filter(self, data):
        """
        Performs a butterworth lowpass filtering
        to remove high frequency noise.
        """
        butt_mtrx = np.empty(shape=data.shape)
        
        num, den = signal.butter(N = self.N, Wn = self.W, btype='low', analog=False)
        e = signal.TransferFunction(num, den, dt=self.Ts)
        
        for col in range(data.shape[1]):
            t_in = np.arange(0,data.shape[0],1)
            t_out, butt_arr = signal.dlsim(e, data[:,col], t=t_in)
            butt_mtrx[:,col] = butt_arr.reshape(-1,1)[:,0]
        
        return butt_mtrx
    
    def _defined_bad_data(self, X, X_cols, y, y_cols):
        """
        For each signal, defines an array of indexes
        whith the value of 0 for the indexes where
        the data is not null and with the value of
        1 for the indexes where the data is null.
        
        A dictionary is defined such data each key
        corresponds to a particular signal and each
        value corresponds to an array of indexes of
        bad data for the corresponding signal.
        
        Arguments:
            X: a matrix of input signals. Each signal is a column;
            X_cols: the input data columns in case they are provided;
            y: a matrix of output signals. Each signal is a column;
            y_cols: the output data columns in case they are provided.
        """
        self.bad_data_dict = defaultdict(list)
        
        #Define input bad data
        for input_idx in range(0, X.shape[1]):
            if X_cols is not None:
                input_idx_name = X_cols[input_idx]
            else:
                input_idx_name = 'input'+'_'+str(input_idx)
            
            self.bad_data_dict[input_idx_name] = np.zeros(len(X[:,input_idx]))  
            self.bad_data_dict[input_idx_name][np.argwhere(np.isnan(X[:,input_idx]))] = 1
            
        #Define output bad data
        for output_idx in range(0, y.shape[1]):
            if y_cols is not None:
                output_idx_name = y_cols[output_idx]
            else:
                output_idx_name = 'output'+'_'+str(output_idx)
            
            self.bad_data_dict[output_idx_name] = np.zeros(len(y[:,output_idx]))  
            self.bad_data_dict[output_idx_name][np.argwhere(np.isnan(y[:,output_idx]))] = 1
    
    def fit_transform(self, X, y):
        """
        Performs all steps in the preprocessing class for the
        given data, which includes:
        
        1) Defining the sequences of bad data (data with null values);
        2) Removing first k samples;
        3) Scaling the data;
        4) Applying a lowpass filter.
        
        Arguments:
            X: the input data in pandas dataframe format or numpy array
            y: the output data in pandas dataframe format or numpy array
        """
        X, y, X_cols, y_cols = self._verify_data(X,y)
        
        #Define Bad Data
        self._defined_bad_data(X=X, X_cols=X_cols, y=y, y_cols=y_cols)
        
        #Remove First Samples from Data
        X_aux = self._remove_first_samples(data=X)
        y_aux = self._remove_first_samples(data=y)
        
        #Scale Data
        data = np.concatenate([X_aux,y_aux],axis=1)
        data = self._scale(data=data)
        
        X_aux = data[:,:X.shape[1]]
        y_aux = data[:,X.shape[1]:]
        
        #Apply Lowpass Filter
        if self.W:
            X_aux = self._lowpass_filter(data=X_aux)
            y_aux = self._lowpass_filter(data=y_aux)
        
        #Create Dataframe
        if X_cols is not None:
            X_aux = pd.DataFrame(X_aux)
            X_aux.columns = X_cols
        
        if y_cols is not None:
            y_aux = pd.DataFrame(y_aux)
            y_aux.columns = y_cols
            
        return X_aux, y_aux