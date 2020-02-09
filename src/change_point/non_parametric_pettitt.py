import pandas as pd
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

class PettittMethod(object):
    """
    Performs signal segmentation using the pettit non-parametric
    method [Pettitt, A.N., 1979. A non-parametric approach to the 
            change-point problem. Appl. Stat. 28, 126–135]
    """
    
    def __init__(self, X, alpha, normalize=True):

        self.df_cols = None
        
        if alpha is None:
            self.alpha = 0.05
        else:
            self.alpha = alpha
        
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
        
        self.segments = None
        self.change_points = None
        self.intervals = None

    def _initialize_segments(self, X, segments):
        for col in range(X.shape[1]):
            segments[col] = [list(range(len(X[:,col])))]

        return segments
    
    def _signal_stat(self, X, segment):
        D = []
        for t in range(len(segment)):
            d_t = np.sign(np.subtract(X[segment[t]],X[segment]))
            D.append(np.sum(d_t))

        return D

    def _mann_whitney(self, D):
        u = 0
        U = []
        for idx in range(len(D)):
            u += D[idx]
            U.append(u)

        return U

    def _cusum_mann_whitney(self, segment, U):
        tau_idx = np.argmax(np.abs(U))
        tau = segment[tau_idx]

        return tau
    
    def _calculate_p_value(self, U):
        K = len(U)
        p_value = 2*np.exp((-6*np.power(np.max(np.abs(U)),2))/(np.power(K,2) + np.power(K,3)))

        return p_value

    def _select_significant_tau(self, p_value_arr, tau_arr, alpha):
        p_args = np.where(np.array(p_value_arr) < alpha)[0]
        if p_args.size > 0:
            sig_tau_arr = list(np.sort(np.array(tau_arr)[p_args]))
            return sig_tau_arr
        else:
            return None
        
    def _update_change_points(self, X, segment_arr):
        p_value_arr = []
        tau_arr = []
        for segment in segment_arr:
            D = self._signal_stat(X, segment)
            U = self._mann_whitney(D)
            tau = self._cusum_mann_whitney(segment,U)
            p_value = self._calculate_p_value(U)
            p_value_arr.append(p_value)
            tau_arr.append(tau)
        return p_value_arr, tau_arr

    def _update_segments(self, sig_tau_arr,N):
        new_segments_arr = []
        for idx in range(len(sig_tau_arr)):
            if len(sig_tau_arr) == 1:
                new_segments_arr.append(list(range(0,sig_tau_arr[idx])))
                new_segments_arr.append(list(range(sig_tau_arr[idx],N+1)))
            elif len(sig_tau_arr) == 2:
                if idx == 0:
                    new_segments_arr.append(list(range(0,sig_tau_arr[idx])))
                else:
                    new_segments_arr.append(list(range(sig_tau_arr[idx-1],sig_tau_arr[idx])))
                    new_segments_arr.append(list(range(sig_tau_arr[idx],N+1)))
            else:
                if idx == 0:
                    new_segments_arr.append(list(range(0,sig_tau_arr[idx])))
                elif idx == len(sig_tau_arr)-1:
                    new_segments_arr.append(list(range(sig_tau_arr[idx-1],sig_tau_arr[idx])))
                    new_segments_arr.append(list(range(sig_tau_arr[idx],N+1)))
                else:
                    new_segments_arr.append(list(range(sig_tau_arr[idx-1],sig_tau_arr[idx])))
        return new_segments_arr
    
    def _create_intervals_from_segments(self):
        X = self.X
        segments = self.segments
        intervals = defaultdict(list)
        
        for col in range(X.shape[1]):
            for segment in segments[col]:
                intervals[col].append([np.min(segment),np.max(segment)])
        
        self.intervals = intervals
        
        return self.intervals
        
    def find_change_points(self):
        X = self.X
        alpha = self.alpha
        self.segments = defaultdict(list)
        self.change_points = defaultdict(list)
        
        #segments initialization
        self.segments = self._initialize_segments(X,self.segments)

        #Find Change-points
        for col in range(X.shape[1]):
            for _ in range(len(X[:,col])):
                p_value_arr, tau_arr = self._update_change_points(X[:,col], self.segments[col])
                sig_tau_arr = self._select_significant_tau(p_value_arr,tau_arr,alpha)
                if sig_tau_arr is not None:
                    self.change_points[col]+=sig_tau_arr
                    self.change_points[col].sort()
                    self.segments[col] = self._update_segments(self.change_points[col],len(X[:,col])-1)
                else:
                    break
        
        intervals = self._create_intervals_from_segments()
        
        return self.intervals, self.change_points

    def plot_change_points(self, show_intervals=False):
        """
        Plots all found change points and its corresponding
        intervals.
        
        Arguments:
            verbose: verbose level as in joblib library
            n_jobs: the number of threads as in joblib library
        """
        
        if self.change_points is None:
            intervals, change_points = self.find_change_points()
        else:
            intervals, change_points = self.intervals, self.change_points
        
        for col in list(intervals.keys()):
            intervals_arr = intervals[col]
            
            sns.set_style("darkgrid")
            plt.figure(figsize=(15,5))
            plt.plot(self.X[:,col])
            if self.df_cols is None:
                col_name = f"Signal {col}"
            else:
                col_name = f"Signal {self.df_cols[col]}"
            plt.title(f"Pettitt Non-Parametric Change Points and Intervals for {col_name}", 
                      fontsize=16, fontweight='bold')
            plt.ylabel("Signal Amplitude", fontsize=14, fontweight='bold')
            plt.xlabel("Discrete Samples", fontsize=14, fontweight='bold')
            plt.xticks(fontsize=14,fontweight='bold',color='grey')
            plt.yticks(fontsize=14,fontweight='bold',color='grey')

            color_rule = True
            color_arr = ['darkmagenta','darkorange']
            
            if show_intervals:
                for interval in intervals_arr:
                    color_rule = not color_rule
                    for idx in interval:
                        plt.scatter(idx, self.X[:,col][idx], marker='X', s=100, color=color_arr[color_rule])
                        plt.axvline(x=idx, linestyle='--', color=color_arr[color_rule])
            else:
                plt.scatter(change_points[col], self.X[change_points[col],col], marker='X', s=100, color='darkmagenta')
                
            plt.show()