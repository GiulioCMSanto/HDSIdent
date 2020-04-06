from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
import pandas as pd
import numpy as np

class Preprocessing(object):
    """
    Performs preprocessing transformations in a given dataframe.
    The following transformations are applied:

        - Data Scaling (Either StandardScaler or MinMaxScaler);
        - Removal of first samples to avoid deflection;
        - Butterworth low-pass filtering.
    
    Arguments:
        df: an input dataframe or array
        scaler: the scaler type to be used
        feature_range: the feature range as a tuple (min, max)
        k: the number of initial samples to be removed
        W: input frequency
        N: Butterworth filter order
        Ts: the sampling frequency of the digital filter (Default = 1.0 seconds)
    """
    def __init__(self, df, scaler='StandardScaler', feature_range = (0,1), k=10, W=0.05, N=1, Ts=1):
        
        if type(df) == pd.core.frame.DataFrame:
            self.df = df.values
            self.df_cols = df.columns
        elif type(df) == np.ndarray:
            self.df = df
            self.df_cols = None
        else:
            raise Exception("Input data must be a pandas dataframe or a numpy array") 
            
        self.scaler = scaler
        self.feature_range = feature_range
        self.k = k
        self.W = W
        self.N = N
        self.Ts = Ts
        
    def scale(self):
        """
        Scales the dataframe according to the scaler type
        provided.
        """
        if self.scaler == 'MinMaxScaler':
            scl = MinMaxScaler(feature_range = self.feature_range)
            self.df = scl.fit_transform(self.df)
        elif self.scaler == 'StandardScaler':
            scl = StandardScaler()
            self.df = scl.fit_transform(self.df)
        else:
            raise Exception("Only MinMaxScaler and StandardScaler are accepted") 
        
        return self.df
    
    def remove_first_samples(self):
        """
        Removes the initial samples to avoid deflection.
        """
        self.df = self.df[self.k:,:]
        
        return self.df
        
    def lowpass_filter(self):
        """
        Performs a butterworth lowpass filtering
        to remove high frequency noise.
        """
        butt_mtrx = np.empty(shape=self.df.shape)
        
        num, den = signal.butter(N = self.N, Wn = self.W, btype='low', analog=False)
        e = signal.TransferFunction(num, den, dt=self.Ts)
        
        for col in range(self.df.shape[1]):
            t_in = np.arange(0,len(self.df),1)
            t_out, butt_arr = signal.dlsim(e, self.df[:,col], t=t_in)
            butt_mtrx[:,col] = butt_arr.reshape(-1,1)[:,0]
        
        self.df = butt_mtrx
        return self.df
    
    def fit_transform(self):
        """
        Performs all steps in the preprocessing class for the
        given dataframe.
        """
        #Scaler
        self.df = self.remove_first_samples()
        self.df = self.scale()
        self.df = self.lowpass_filter()
        
        if self.df_cols is not None:
            self.df = pd.DataFrame(self.df)
            self.df.columns = self.df_cols
            
        return self.df