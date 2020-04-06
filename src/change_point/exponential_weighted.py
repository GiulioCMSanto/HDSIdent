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
        
        #Internal Variables
        self.unified_intervals = defaultdict(list)
        self.intervals = defaultdict(list)
        self._mu_k_arr = None
        self._v_k_arr = None
        self._mu_k = np.array([])
        self._v_k = np.array([])
        self._mu_k_1 = 0
        self._v_k_1 = 0
        self._is_interval = [False]*self.X.shape[1]
        self._init_idx = [0]*self.X.shape[1]
        self._final_idx = [0]*self.X.shape[1]
        self._criteria = None
        self._intervals_parameter_grid = {}
        
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
        
        self._mu_k_1 = 0
        self._v_k_1 = 0
        
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
        
        #Reset Internal Variables
        self.unified_intervals = defaultdict(list)
        self.intervals = defaultdict(list)
        self._mu_k_arr = None
        self._v_k_arr = None
        self._mu_k = np.array([])
        self._v_k = np.array([])
        self._mu_k_1 = 0
        self._v_k_1 = 0
        self._is_interval = [False]*self.X.shape[1]
        self._init_idx = [0]*self.X.shape[1]
        self._final_idx = [0]*self.X.shape[1]
        self._criteria = None
        
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
    
    def plot_filters_grid(self):
        """
        Plots several combinations of forgetting factor parameters to help tunning.
        """
        
        #Save Original Forgetting Factors
        original_forgetting_fact_v = self.forgetting_fact_v
        original_forgetting_fact_u = self.forgetting_fact_u
        
        #Grid of forgetting factors
        param_grid = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01]
        
        #Plot Grid
        plot_arr = []
        param_arr = []
        for idx, param in enumerate(param_grid):
            self.forgetting_fact_v = param
            self.forgetting_fact_u = param
            mu_k_arr, v_k_arr = self.recursive_exponential_moving_average_and_variance()
            plot_arr.append(v_k_arr)
            plot_arr.append(mu_k_arr)
            param_arr.append(param)
            param_arr.append(param)
        
        for col in range(self.X.shape[1]):
            if self.df_cols is not None:
                col_name = self.df_cols[col]
            else:
                col_name = col
                
            print("Grid for Column: {}".format(col_name))
            
            plt.figure(figsize=(15,60))
            for idx, element in enumerate(plot_arr):
                plt.subplot(len(plot_arr),2,idx+1)
                if (idx+1)%2 == 0:
                    plt.plot(element[:,col], color='darkmagenta', linewidth=2)
                    plt.title("{}: Moving Average with Lambda = {}".format(col_name,param_arr[idx]),
                              fontweight='bold',fontsize=14)
                else:
                    plt.plot(element[:,col], color='darkorange', linewidth=2)
                    plt.title("{}: Moving Variance with Lambda = {}".format(col_name,param_arr[idx]),
                              fontweight='bold',fontsize=14)
            plt.tight_layout()
            plt.show()
        
        self.forgetting_fact_v = original_forgetting_fact_v
        self.forgetting_fact_u = original_forgetting_fact_u
    
    def _plot_intervals_grid_surface(self,
                                     plot_title,
                                     x_axis,
                                     x_label,
                                     y_axis,
                                     y_label,
                                     z_axis,
                                     z_label):
        """
        This function creates a 3D Surface Plot containing the number
        of intervals produced by the moving average algorithm for a
        grid of parameters. The figure is produced using Plotly.
        
        Arguments:
            plot_title: the surface plot title.
            x_axis: the x axis.
            x_label: the x axis label.
            y_axis: the y axis.
            y_label: the y axis label.
            z_axis: the z axis.
            z_label: the x axis label.
        """
        fig = go.Figure(go.Surface(
            x = x_axis,
            y = y_axis,
            z = z_axis,
            colorscale='PuOr'
            ))

        fig.update_layout(autosize=True,
                          width=1000,
                          height=500,
                          scene = dict(
                            xaxis = dict(title = x_label, nticks = 10),
                            yaxis = dict(title = y_label, nticks = 10),
                            zaxis = dict(title = z_label, nticks = 10),
                            aspectratio =  {"x": 1.2, "y": 1.2, "z": 0.5},
                            camera = dict(eye=dict(x=1.2, y=1.2, z=0.1))
                          ),
                          title={
                          'text': plot_title,
                          'y':0.9,
                          'x':0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'}
                        )
        
        fig.show()
    
    def _intervals_grid_iteration(self, factor_grid, H_v_grid, individual_plot):
        """
        This function iterates over the forgetting factor parameters and the
        threshold parameters to create the parameters grid.
        
        Arguments:
            factor_grid: the forgetting factor values used to create the parameters grid.
            H_v_grid: the variance threshold values used to create the parameters grid.
            individual_plot: If True, a 3D surface plot will be produce
            for each signal in the dataframe. If False, all signals will
            have the same value of forgetting factors and will be evaluated
            under the same variance threshold.  
        """
        
        #Determines the number of signals to set the Variance threshold
        if individual_plot:
            k = 1
        else:
            k = self.X.shape[1]
        
        #Compute grid matrix
        num_intervals_arr = np.zeros((len(factor_grid),len(H_v_grid)))
        for idx_1, factor in enumerate(factor_grid):
            for idx_2, H_v in enumerate(H_v_grid):
                self.forgetting_fact_v = factor
                self.forgetting_fact_u = factor
                self.H_v = [H_v]*k

                intervals = self.fit()
                num_intervals_arr[idx_2,idx_1] = len(intervals.keys())
        
        return num_intervals_arr
    
    def _create_intervals_parameters_grid_df(self, num_intervals_arr, factor_grid, H_v_grid, col):
        """
        Creates a dictinary of dataframes for the grid of parameters vs intervals,
        for the given signal.
        
        Arguments:
            num_intervals_arr: the parameters grid matrix, where each row corresponds
            to a forgetting factor and each column corresponds to a variance threshold.
            factor_grid: the forgetting factor values used to create the parameters grid.
            H_v_grid: the variance threshold values used to create the parameters grid.
            col: the column index of the signal being considered in the grid.
        """
        
        #Define the dictionary key
        if col is None:
            key = 'all_signals'
        else:
            key = col
        
        #Create a dataframe based on the num_intervals_arr matrix
        self._intervals_parameter_grid[key] = pd.DataFrame(num_intervals_arr)
        self._intervals_parameter_grid[key].index = factor_grid
        self._intervals_parameter_grid[key].columns = H_v_grid
        
        #Update the dataframe index and column names
        df = {}
        df['Variance Threshold'] = self._intervals_parameter_grid[key]
        self._intervals_parameter_grid[key] = pd.concat(df,axis=1)
        self._intervals_parameter_grid[key] = self._intervals_parameter_grid[key].rename_axis(['Forgetting Factor'])
        
        #Inser Color in the Dataframe
        cm = sns.light_palette("green", as_cmap=True)
        self._intervals_parameter_grid[key] = self._intervals_parameter_grid[key].style.background_gradient(cmap=cm)
        
    def plot_intervals_grid(self, individual_plot = False):
        """
        This function plots a 3D surface figure containing the number
        of intervals produced by the moving average algorithm for a
        grid of parameters.
        
        Arguments:
            individual_plot: If True, a 3D surface plot will be produce
            for each signal in the dataframe. If False, all signals will
            have the same value of forgetting factors and will be evaluated
            under the same variance threshold.
        """
        
        #Save Original Forgetting Factors
        original_forgetting_fact_v = self.forgetting_fact_v
        original_forgetting_fact_u = self.forgetting_fact_u
        original_H_v = self.H_v
        original_X = self.X
        
        #Grid of forgetting factors
        factor_grid = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01]
        
        #Grid of Thresholds
        H_v_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1]
        
        #Iterate Over Grid
        if individual_plot:
            for col in range(self.X.shape[1]):
                
                #Take Corresponding Signal of index col
                self.X = self.X[:,col].reshape(-1,1)
                
                #Take Signal Name
                if self.df_cols is not None:
                    col_name = self.df_cols[col]
                else:
                    col_name = col
                
                #Take Grid of Parameters
                num_intervals_arr = self._intervals_grid_iteration(factor_grid = factor_grid,
                                                                   H_v_grid = H_v_grid,
                                                                   individual_plot = individual_plot)
                
                #Create the Grid Dataframe
                self._create_intervals_parameters_grid_df(num_intervals_arr = num_intervals_arr,
                                                          factor_grid = factor_grid,
                                                          H_v_grid = H_v_grid,
                                                          col = col_name)
            
                plot_title = 'Number of Intervals X Parameters for Signal {}'.format(col_name)
                x_label = 'Forgetting Factor'
                y_label = 'Variance Threshold'
                z_label = 'Number of Intervals'
                self._plot_intervals_grid_surface(plot_title=plot_title,
                                                  x_axis=factor_grid,
                                                  x_label=x_label,
                                                  y_axis=H_v_grid,
                                                  y_label=y_label,
                                                  z_axis=num_intervals_arr,
                                                  z_label=z_label)
                self.X = original_X
        else:
            num_intervals_arr = self._intervals_grid_iteration(factor_grid = factor_grid,
                                                               H_v_grid = H_v_grid,
                                                               individual_plot = individual_plot)
            
            self._create_intervals_parameters_grid_df(num_intervals_arr = num_intervals_arr,
                                                      factor_grid = factor_grid,
                                                      H_v_grid = H_v_grid,
                                                      col = None)
            
            plot_title = 'Number of Intervals X Parameters for All Signals'
            x_label = 'Forgetting Factor'
            y_label = 'Variance Threshold'
            z_label = 'Number of Intervals'
            self._plot_intervals_grid_surface(plot_title=plot_title,
                                              x_axis=factor_grid,
                                              x_label=x_label,
                                              y_axis=H_v_grid,
                                              y_label=y_label,
                                              z_axis=num_intervals_arr,
                                              z_label=z_label)
        
        #Reset Class Input variables
        self.forgetting_fact_v = original_forgetting_fact_v
        self.forgetting_fact_u = original_forgetting_fact_u
        self.H_v = original_H_v 
        self.X = original_X
            
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
                plt.plot(self._v_k_arr[:,col])
            elif self._criteria == 'average':
                plt.plot(self._mu_k_arr[:,col])
            else:
                plt.plot(self._v_k_arr[:,col],color='blue',label='Variance Plot')
                plt.plot(self._mu_k_arr[:,col],color='brown',label='Average Plot')
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
            color_arr = ['darkmagenta','darkorange']
            for interval in intervals_arr:
                color_rule = not color_rule
                for idx in interval:
                    if self._criteria == 'variance':
                        plt.scatter(idx, self._v_k_arr[:,col][idx], marker='X', s=100, color=color_arr[color_rule])
                        plt.axvline(x=idx, linestyle='--', color=color_arr[color_rule])
                    else:
                        plt.scatter(idx, self._mu_k_arr[:,col][idx], marker='X', s=100, color=color_arr[color_rule])
                        plt.axvline(x=idx, linestyle='--', color=color_arr[color_rule])
            plt.show()