import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

class ExponentialWeighted(object):
    """
    Exponential Moving Average Control Chart. Performs a
    recursive moving average filter and detects
    change points and its corresponding intervals.

    Arguments:
        X: the input discrete-time data
        forgetting_fact_v: exponential forgetting factor for the variance
        forgetting_fact_u: exponential forgetting factor for the average
        sigma: data (population) standard deviation
        H_u: change-point threshold for the mean
        H_v: change-point threshold for the variance
        normalize: whether or not to normalized the data (StandardScaler)
        verbose: verbose level as in joblib library
        n_jobs: the number of threads as in joblib library
    """
    def __init__(self, X, forgetting_fact_v, forgetting_fact_u, sigma = None, H_u=None, H_v = None, n_jobs=-1, verbose=0):

        self.forgetting_fact_v = forgetting_fact_v
        self.forgetting_fact_u = forgetting_fact_u
        self.df_cols = None
        self.n_jobs = n_jobs
        self.verbose = verbose

        if type(X) == pd.core.frame.DataFrame:
            self.X = X.values
            self.df_cols = X.columns
        elif type(X) == np.ndarray:
            self.X = X
        else:
            raise Exception("Input data must be a pandas dataframe or a numpy array") 
            
        if not sigma:
            self.sigma = np.std(X,axis=0)
        else:
            self.sigma = sigma
        
        if not H_u:
            self.H_u = 5*self.sigma
        else:
            self.H_u = H_u
            
        if not H_v:
            self.H_v = 5*self.sigma
        else:
            self.H_v = H_v
        
    def _initialize_internal_variables(self):
        """
        This function initializes the interval variables.
        """
        self.unified_intervals = defaultdict(list)
        self.intervals = defaultdict(list)
        self._mu_k_arr = None
        self._v_k_arr = None
        self._mu_k = np.array([])
        self._v_k = np.array([])
        self._is_interval = [False]*self.X.shape[1]
        self._init_idx = [0]*self.X.shape[1]
        self._final_idx = [0]*self.X.shape[1]
        self._criteria = None
        
        self._mu_k_1 = np.mean(self.X[:100,:],axis=0)
        self._v_k_1 = np.var(self.X[:100,:],axis=0)

        
    def _exponential_moving_average_and_variance(self, X, idx):
        """
        Performs a recursive exponential moving average/variance 
        algorithm from past samples.
        
        Arguments:
            X: the input discrete-time data
            idx: the input data sample index
        
        Output:
            self._mu_k: the sample average filtered data for the given index
            self._v_k: the sample variance filtered data for the given index
        """
        
        self._mu_k = self.forgetting_fact_u*X[idx,:] + (1-self.forgetting_fact_u)*self._mu_k_1
        self._v_k = ((2-self.forgetting_fact_u)/2)*(self.forgetting_fact_v*(X[idx,:]-self._mu_k)**2 + 
                                                   (1-self.forgetting_fact_v)*self._v_k_1)
        
        self._mu_k_1 = self._mu_k
        self._v_k_1 = self._v_k
        
        return (self._mu_k, self._v_k)
    
    def _search_for_change_points(self, idx, col, criteria):
        """
        Searchs for change points in the filtered data.
        
        Arguments:
            idx: the filtered data sample index
            col: the data column (execution signal)
            criteria: the filter to be considered when looking for
                      a change-point (average, variance or both)
        
        Output:
            self._intervals: a list with the initial and final
                             indexes of an interval (if found).
        """
            
        #Change-point conditions
        if criteria == 'average':
            condition = abs(self._mu_k_arr[idx,col]) >= self.H_u[col]
        elif criteria == 'variance':
            condition = abs(self._v_k_arr[idx,col]) >= self.H_v[col]
        else:
            condition = (abs(self._mu_k_arr[idx,col]) >= self.H_u[col]) and \
                        (abs(self._v_k_arr[idx,col]) >= self.H_v[col])

        if condition:
            if not self._is_interval[col]:
                self._init_idx[col] = idx
                self._is_interval[col] = True
            elif idx == len(self.X)-1 and self._is_interval[col]:
                self._is_interval[col] = False
                self._final_idx[col] = idx
                self.intervals[col].append([self._init_idx[col], self._final_idx[col]])    
        elif self._is_interval[col]:
            self._is_interval[col] = False
            self._final_idx[col] = idx
            self.intervals[col].append([self._init_idx[col], self._final_idx[col]])
        
    def recursive_exponential_moving_average_and_variance(self):
        """
        Performs a recursive moving average/variance algorithm from past samples
        using a multithread approach.
        
        Output:
            self._mu_k_arr: the average filtered data for the given index
            self._v_k_arr: the variance filtered data for the given index
        """
        X = self.X
        results = list(Parallel(n_jobs=self.n_jobs,
                                require='sharedmem',
                                verbose=self.verbose)(delayed(self._exponential_moving_average_and_variance)(X, idx)
                                                      for idx in range(len(X))))
         
        self._mu_k_arr, self._v_k_arr = list(zip(*results))
        self._mu_k_arr = np.stack(self._mu_k_arr,axis=0)
        self._v_k_arr = np.stack(self._v_k_arr,axis=0)
        
        return self._mu_k_arr, self._v_k_arr
    
    def change_points(self, criteria='variance'):
        """
        Searchs for change points in the filtered data and its
        corresponding intervals using a multithread approach.
        
        Arguments:
            criteria: the filter to be considered when looking for
                      a change-point (average, variance or both)
        """
        #Reset Intervals
        self.intervals = defaultdict(list)
        
        #Update Criteria
        self._criteria = criteria
        
        if (self._mu_k_arr is None) or (self._v_k_arr is None):
            self._mu_k_arr, self._v_k_arr = self.recursive_exponential_moving_average_and_variance()
            
        Parallel(n_jobs=self.n_jobs,
                 require='sharedmem',
                 verbose=self.verbose)(delayed(self._search_for_change_points)(idx,col,criteria)
                                       for idx in range(len(self.X))
                                       for col in range(self.X.shape[1]))
        
        self._is_interval = [False]*self.X.shape[1]
        self._init_idx = [0]*self.X.shape[1]
        self._final_idx = [0]*self.X.shape[1]
        
        return self.intervals

    
    def _create_indicating_sequence(self):
        """
        This function creates an indicating sequence, i.e., an array containing 1's
        in the intervals of interest and 0's otherwise, based on each interval obtained
        by the bandpass filter approach.
        
        Output:
            indicating_sequence: the indicating sequence
        """
        indicating_sequence = np.zeros(self.X.shape[0])
        for _, interval_arr in self.intervals.items():
            for interval in interval_arr:
                indicating_sequence[interval[0]:interval[1]+1] = 1
                
        return indicating_sequence

    def _define_intervals_from_indicating_sequence(self,indicating_sequence):
        """
        Receives an indicating sequence, i.e., an array containing 1's
        in the intervals of interest and 0's otherwise, and create a
        dictionary with the intervals indexes.

        Arguments:
            indicating_sequence: an array containing 1's in the intervals of
            interest and 0's otherwise.

        Output:
            sequences_dict: a dictionary with the sequences indexes
        """
        is_interval = False
        interval_idx = -1

        sequences_dict = defaultdict(list)
        for idx, value in enumerate(indicating_sequence):
            if idx == 0 and value == 1:
                is_interval = True
                interval_idx += 1
            elif idx > 0:
                if value == 1 and indicating_sequence[idx-1] == 0 and not is_interval:
                    is_interval = True
                    interval_idx += 1
                elif value == 0 and indicating_sequence[idx-1] == 1 and is_interval:
                    is_interval = False

            if is_interval:
                sequences_dict[interval_idx].append(idx)

        return sequences_dict
    
    def fit(self):
        """
        This function performs the following routines:
            - Applies the recursive exponential moving average/variance
            - Compute the initial intervals (change-points)
            - Creates an indicating sequence, unifying input and output intervals
            - From the indicating sequence, creates a final unified interval
        
        Output:
            unified_intervals: the final unified intervals for the input and output signals
        """
        
        #Initialize Internal Variables
        self._initialize_internal_variables()
        
        #Apply Recursive Exponential Moving Average/Variance
        self._mu_k_arr, self._v_k_arr = self.recursive_exponential_moving_average_and_variance()
        
        #Find change-points
        _ = self.change_points()
        
        #Create Indicating Sequence
        indicating_sequence = self._create_indicating_sequence()
        
        #Define Intervals
        self.unified_intervals = self._define_intervals_from_indicating_sequence(indicating_sequence=
                                                                                 indicating_sequence)
            
        return self.unified_intervals
            
    def plot_change_points(self):
        """
        Plots all found change points and its corresponding
        intervals.
        """
        if (self._mu_k_arr is None) or (self._v_k_arr is None):
            self._mu_k_arr, self._v_k_arr = self.recursive_exponential_moving_average_and_variance()
        
        if not self.intervals:
            intervals = self.change_points()
        else:
            intervals = self.intervals
        
        for col in list(intervals.keys()):
            intervals_arr = intervals[col]
            
            sns.set_style("darkgrid")
            plt.figure(figsize=(15,5))
            if self._criteria == 'variance':
                plt.plot(self._v_k_arr[:,col], zorder=1, color='coral')
            elif self._criteria == 'average':
                plt.plot(self._mu_k_arr[:,col], zorder=1, color='coral')
            else:
                plt.plot(self._v_k_arr[:,col],color='blue',label='Variance Plot', zorder=1, color='coral')
                plt.plot(self._mu_k_arr[:,col],color='brown',label='Average Plot', zorder=1, color='coral')
                plt.legend(fontsize=14)
            
            if self.df_cols is None:
                col_name = f"Signal {col}"
            else:
                col_name = f"Signal {self.df_cols[col]}"
            plt.title(f"Moving Average Change Points and Intervals for {col_name}", fontsize=18, fontweight='bold')
            plt.ylabel("Signal Amplitude", fontsize=18, fontweight='bold')
            plt.xlabel("Discrete Samples", fontsize=18, fontweight='bold')
            plt.xticks(fontsize=18,fontweight='bold',color='grey')
            plt.yticks(fontsize=18,fontweight='bold',color='grey')

            color_rule = True
            color_arr = ['darkred','darkmagenta']
            for interval in intervals_arr:
                color_rule = not color_rule
                for idx in interval:
                    if self._criteria == 'variance':
                        plt.scatter(idx, self._v_k_arr[:,col][idx], marker='x', s=50, color=color_arr[color_rule], zorder=2)
                        plt.axvline(x=idx, linestyle='--', color=color_arr[color_rule])
                    else:
                        plt.scatter(idx, self._mu_k_arr[:,col][idx], marker='x', s=50, color=color_arr[color_rule], zorder=2)
                        plt.axvline(x=idx, linestyle='--', color=color_arr[color_rule])
            plt.show()