import pandas as pd
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

class moving_average(object):
    """
    Moving Average Control Chart. Performs a
    recursive moving average filter and detects
    change points and its corresponding intervals.
    """
    def __init__(self, X, W, sigma = None, H=None, normalize=True):
        """ 
        Constructor.
        
        Arguments:
            X: the input discrete-time data
            W: the window length
            sigma: data (population) standard deviation
            H: change-point threshold
            normalize: whether or not to normalized the data (StandardScaler)
        """
        self.W = W
        self.df_cols = None

        if type(X) == pd.core.frame.DataFrame:
            self.X = X.values
            self.df_cols = X.columns
        elif type(X) == np.ndarray:
            self.X = X
        else:
            raise Exception("Input data must be a pandas dataframe or a numpy array") 
        
        if normalize:
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
            
        if not sigma:
            self.sigma = np.std(X,axis=0)
        else:
            self.sigma = sigma
        
        if not H:
            self.H = 5*self.sigma
        else:
            self.H = H
        
        self.mu_arr = None
        self.intervals = defaultdict(list)
        
        #Internal Variables
        self._mu=np.array([])
        self._is_interval = [False]*X.shape[1]
        self._init_idx = [0]*X.shape[1]
        self._final_idx = [0]*X.shape[1]
        
    def _moving_average(self, X, W, idx):
        """
        Performs a recursive moving average algorithm from past samples.
        If past samples do not exist in time yet, the window length will 
        have the same size as the number of available samples.
        
        Arguments:
            X: the input discrete-time data
            W: the window length
            idx: the input data sample index
        
        Output:
            self._mu: the sample filtered data for the given index
        """
        if idx-W+1 <= 0: #Data lenght smaller than window length
            if idx == 0: #First index
                self._mu = X[idx,:]
            else:
                self._mu = np.sum(X[0:idx+1,:],axis=0)/len(X[:idx+1,:])
        else:
            self._mu = np.add(self._mu,np.subtract(X[idx,:],X[idx-W,:])/W) #Recursive approach
        
        return self._mu

    def _search_for_change_points(self, mu_arr, idx, H):
        """
        Searchs for change points in the filtered data.
        
        Arguments:
            mu_arr: the filtered data
            idx: the filtered data sample index
            H: the change-point threshold
        
        Output:
            self._intervals: a list with the initial and final
                             indexes of an interval (if found).
        """
        for col in range(mu_arr.shape[1]):
            if abs(mu_arr[idx,col]) >= H[col]:
                if not self._is_interval[col]:
                    self._init_idx[col] = idx
                    self._is_interval[col] = True
                elif idx == len(self.X)-1 and self._is_interval[col]:
                    self._is_interval[col] = False
                    self._final_idx[col] = idx
                    self.intervals[col].append([self._init_idx[col], self._final_idx[col]])    
            elif self._is_interval[col]:
                self._is_interval[col] = False
                self._final_idx[col] = idx-1
                self.intervals[col].append([self._init_idx[col], self._final_idx[col]])
        
    def recursive_moving_average(self, verbose=10, n_jobs=-1):
        """
        Performs a recursive moving average algorithm from past samples
        using a multithread approach.
        
        Arguments:
            verbose: verbose level as in joblib library
            n_jobs: the number of threads as in joblib library
        """
        X = self.X
        W = self.W
        self.mu_arr = list(Parallel(n_jobs=n_jobs,
                           require='sharedmem',
                           verbose=verbose)(delayed(self._moving_average)(X, W, idx)
                                            for idx in range(len(X))))
 
        self.mu_arr = np.stack(self.mu_arr,axis=0)
        self._mu = np.array([])
        
        return self.mu_arr
    
    def chenge_points(self, verbose=10, n_jobs=-1):
        """
        Searchs for change points in the filtered data and its
        corresponding intervals using a multithread approach.
        
        Arguments:
            verbose: verbose level as in joblib library
            n_jobs: the number of threads as in joblib library
        """
        H = self.H
        
        if self.mu_arr is None:
            mu_arr = self.recursive_moving_average(verbose, n_jobs)
        else:
            mu_arr = self.mu_arr
            
        Parallel(n_jobs=n_jobs,
                 require='sharedmem',
                 verbose=verbose)(delayed(self._search_for_change_points)(mu_arr,idx,H)
                                  for idx in range(len(mu_arr)))
        
        self._is_interval = [False]*self.X.shape[1]
        self._init_idx = [0]*self.X.shape[1]
        self._final_idx = [0]*self.X.shape[1]
        
        return self.intervals
    
    def plot_change_points(self, verbose=10, n_jobs=-1):
        """
        Plots all found change points and its corresponding
        intervals.
        
        Arguments:
            verbose: verbose level as in joblib library
            n_jobs: the number of threads as in joblib library
        """
        if self.mu_arr is None:
            mu_arr = self.recursive_moving_average(verbose=verbose, n_jobs=n_jobs)
        else:
            mu_arr = self.mu_arr
        
        if not self.intervals:
            intervals = self.chenge_points(verbose=verbose, n_jobs=n_jobs)
        else:
            intervals = self.intervals
        
        for col in list(intervals.keys()):
            intervals_arr = intervals[col]
            
            sns.set_style("darkgrid")
            plt.figure(figsize=(15,5))
            plt.plot(mu_arr[:,col])
            if self.df_cols is None:
                col_name = f"Signal {col}"
            else:
                col_name = f"Signal {self.df_cols[col]}"
            plt.title(f"Moving Average Change Points and Intervals for {col_name}", 
                      fontsize=16, fontweight='bold')
            plt.ylabel("Signal Amplitude", fontsize=14, fontweight='bold')
            plt.xlabel("Discrete Samples", fontsize=14, fontweight='bold')
            plt.xticks(fontsize=14,fontweight='bold',color='grey')
            plt.yticks(fontsize=14,fontweight='bold',color='grey')

            color_rule = True
            color_arr = ['darkmagenta','darkorange']
            for interval in intervals_arr:
                color_rule = not color_rule
                for idx in interval:
                    plt.scatter(idx, mu_arr[:,col][idx], marker='X', s=100, color=color_arr[color_rule])
                    plt.axvline(x=idx, linestyle='--', color=color_arr[color_rule])
            plt.show()