import pandas as pd
import numpy as np
from scipy import signal
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

class cusum():
    
    def __init__(self, X, mu = None, mu1 = None, sigma = None, n_sigmas = None, H = None, K = None):
        
        self.X = X
        
        if not mu:
            self.mu = np.mean(self.X)
        else:
            self.mu = mu
        
        if not sigma:
            self.sigma = np.std(self.X, ddof=1)
        else:
            self.sigma = sigma
        
        if not n_sigmas:
            self.n_sigmas = 1
        else:
            self.n_sigmas = n_sigmas
            
        if not mu1:
            self.mu1 = self.mu + self.n_sigmas*self.sigma
        else:
            self.mu1 = mu1
        
        if not K:
            self.K = np.abs(self.mu1 - self.mu)/2
        else:
            self.K = K
        
        if not H:
            self.H = 5*self.sigma
        else:
            self.H = H
    
    def tabular_cumsum(self):
        X = self.X
        K = self.K
        mu = self.mu
        
        Cpp, Cp = 0, list()
        Cll, Cl = 0, list()
        for idx in range(len(X)):
            Cpp = np.maximum(0, X[idx] - (mu+K) + Cpp)
            Cll = np.maximum(0, (mu-K) - X[idx] + Cll)
            Cp.append(Cpp)
            Cl.append(Cll)

        return Cp, Cl
    
    def standardized_cumsum(self):
        X = self.X
        K = self.K
        sigma = self.sigma
        mu = self.mu
        
        y = (X-mu)/sigma
        
        Cpp, Cp = 0, list()
        Cll, Cl = 0, list()
        
        for idx in range(len(X)):
            Cpp = np.maximum(0, y[idx] - K + Cpp)
            Cll = np.maximum(0, - K - y[idx] + Cll)
            Cp.append(Cpp)
            Cl.append(Cll)
            
        return Cp, Cl   
    
    def variability_cumsum(self):
        X = self.X
        K = self.K
        sigma = self.sigma
        mu = self.mu
        
        y = (X-mu)/sigma
        v = (np.sqrt(np.abs(y))-0.822)/0.349
        
        Spp, Sp = 0, list()
        Sll, Sl = 0, list()
        
        for idx in range(len(X)):
            Spp = np.maximum(0, v[idx] - K + Spp)
            Sll = np.maximum(0, -K - v[idx] + Sll)
            Sl.append(Sll)
            Sp.append(Spp)
        return Sp, Sl   
    
    def change_points(self):
        init_idx = 0
        final_idx = 0
        intervals = list()
        is_interval = False
        Cp, Cl = self.tabular_cumsum()
        H = self.H
        
        for idx in range(len(Cp)):
            if (Cp[idx] > H or Cl[idx] > H):
                if not is_interval:
                    init_idx = idx
                is_interval = True
            elif is_interval:
                final_idx = idx-1
                intervals.append([init_idx, final_idx])
                is_interval = False
        
        return intervals
        
class moving_average():
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

class bandpass_filter():
    """
    Performs signal segmentation using a discrete-time Butterworth
    bandpass filter from SciPy. Notice that the input frequencies are
    normalized into 0 and 1.
    """
    def __init__(self, X, W, N, Ts = None, sigma = None, H=None, normalize=True):
        """ 
        Constructor.

        Arguments:
            X: the input discrete-time data
            W: input frequency [W1, W2] array as in scipy documentation
            N: Butterworth filter order
            sigma: data (population) standard deviation
            H: change-point threshold
            Ts: the sampling frequency of the digital filter (Default = 1.0 seconds)
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
        
        if not Ts:
            self.Ts = 1
        else:
            self.Ts = Ts
        
        if not N:
            self.N = 1
        else:
            self.N = N
        
        self.intervals = None
        
        #Internal Variables
        self.butt_mtrx = None
        self._is_interval = [False]*self.X.shape[1]
        self._init_idx = [0]*self.X.shape[1]
        self._final_idx = [0]*self.X.shape[1]
        self._err_points = list()
        self._num = 0
        self._den = 0
        
    def butterworth_filter(self):
        """
        Apply a Butterworth bandpass filter to the input data.
        
        Output:
            butt_mtrx: the filtered data
        """
        X = self.X
        N = self.N
        W = self.W
        Ts = self.Ts
        self.butt_mtrx = np.empty(shape=self.X.shape)
        
        self.num, self.den = signal.butter(N = N, Wn = W, fs=1/Ts, btype='bandpass', analog=False)
        e = signal.TransferFunction(self.num, self.den, dt=Ts)
        
        for col in range(X.shape[1]):
            t_in = np.arange(0,len(X),1)
            t_out, butt_arr = signal.dlsim(e, X[:,col], t=t_in)
            self.butt_mtrx[:,col] = butt_arr.reshape(-1,1)[:,0]
        
        return self.butt_mtrx
    
    def _search_for_change_points(self, err_points, idx):
        """
        Receives the errors array and use it to find change points and
        its corresponding intervals.
        
        Arguments:
            err_points: an array with error values for camparing with a threshold
            idx: array index
        """
        for col in range(self.X.shape[1]):
            if idx in list(err_points[col]):
                if not self._is_interval[col]:
                    self._init_idx[col] = idx
                    self._is_interval[col] = True
                elif idx == len(self.X)-1 and self._is_interval[col]:
                    self._is_interval[col] = False
                    self._final_idx[col] = idx
                    self.intervals[col].append([self._init_idx[col], self._final_idx[col]])  
            elif self._is_interval[col]:
                self._final_idx[col] = idx-1
                self.intervals[col].append([self._init_idx[col], self._final_idx[col]])
                self._is_interval[col] = False
    
            
    def chenge_points(self, verbose=10, n_jobs=-1):
        """
        Computes an error array and use it for finding
        changing points based on a given threshold H.
        
        Arguments:
            verbose: verbose level as in joblib library
            n_jobs: the number of threads as in joblib library
        """
        X = self.X
        H = self.H
        self.intervals = defaultdict(list) 
        
        if self.butt_mtrx is None:
            self.butt_mtrx = self.butterworth_filter()
        else:
            butt_mtrx = self.butt_mtrx
        
        for col in range(X.shape[1]):
            self._err_points.append(np.where(np.abs(butt_mtrx[:,col]) >= H[col])[0])

        Parallel(n_jobs=n_jobs,
                 require='sharedmem',
                 verbose=verbose)(delayed(self._search_for_change_points)(self._err_points,idx)
                                  for idx in range(len(X)))
        
        return self.intervals
    
    def bode_plot(self):
        """
        Plots the Butterworth discrete-time filter.
        """
        np.seterr(divide = 'ignore')
        sns.set_style("darkgrid")
        w, h = signal.freqz(self.num,self.den)
        fig, ax1 = plt.subplots(figsize=(15,5))
        ax1.set_title('Digital Filter Frequency Response', fontsize=16, fontweight='bold')
        ax1.plot(w, 20 * np.log10(abs(h)), 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Frequency [rad/sample]', fontsize=14, fontweight='bold')
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g', fontsize=14, fontweight='bold')
        ax2.grid()
        ax2.axis('tight')
        plt.show()  
        
    def plot_change_points(self, verbose=10, n_jobs=-1):
        """
        Plots all found change points and its corresponding
        intervals.
        
        Arguments:
            verbose: verbose level as in joblib library
            n_jobs: the number of threads as in joblib library
        """
        
        if self.butt_mtrx is None:
            butt_mtrx = self.butterworth_filter()
        else:
            butt_mtrx = self.butt_mtrx
        
        if self.intervals is None:
            intervals = self.chenge_points()
        else:
            intervals = self.intervals
        
        for col in range(len(intervals.keys())):
            intervals_arr = intervals[col]
            X = self.X
            H = self.H
            
            sns.set_style("darkgrid")
            plt.figure(figsize=(15,5))
            plt.plot(butt_mtrx[:,col])
            plt.plot([H[col]]*len(X),color='gray')
            plt.annotate(f'+H = {H[col]}',
                         xy=(len(X)-40, np.around(H[col]+0.15,2)),
                         fontsize=12,
                         fontweight='bold',
                         color='black')
            plt.plot([-H[col]]*len(X),color='gray')
            plt.annotate(f'-H = {-H[col]}',
                         xy=(len(X)-40, np.around(-H[col]-0.25,2)),
                         fontsize=12,
                         fontweight='bold',
                         color='black')
            if self.df_cols is None:
                col_name = f"Signal {col}"
            else:
                col_name = f"Signal {self.df_cols[col]}"
            plt.title(f"Bandpass Filter Change Points and Intervals for {col_name}", fontsize=16, fontweight='bold')
            plt.ylabel("Signal Amplitude", fontsize=14, fontweight='bold')
            plt.xlabel("Discrete Samples", fontsize=14, fontweight='bold')
            plt.xticks(fontsize=14,fontweight='bold',color='grey')
            plt.yticks(fontsize=14,fontweight='bold',color='grey')
            color_rule = True
            color_arr = ['darkmagenta','darkorange']
            for interval in intervals_arr:
                color_rule = not color_rule
                for idx in interval:
                    plt.scatter(idx, butt_mtrx[:,col][idx], marker='X', s=100, color=color_arr[color_rule])
                    plt.axvline(x=idx, linestyle='--', color=color_arr[color_rule])
            plt.show()

class non_parametric_pettitt():
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
    
class exponential_moving_average_and_variance():
    """
    Exponential Moving Average Control Chart. Performs a
    recursive moving average filter and detects
    change points and its corresponding intervals.
    """
    def __init__(self, X, forgetting_fact_v, forgetting_fact_u, sigma = None, H_u=None, H_v = None, normalize=True):
        """ 
        Constructor.
        
        Arguments:
            X: the input discrete-time data
            forgetting_fact_v: exponential forgetting factor for the variance
            forgetting_fact_u: exponential forgetting factor for the average
            sigma: data (population) standard deviation
            H_u: change-point threshold for the mean
            H_v: change-point threshold for the variance
            normalize: whether or not to normalized the data (StandardScaler)
        """
        self.forgetting_fact_v = forgetting_fact_v
        self.forgetting_fact_u = forgetting_fact_u
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
        
        if not H_u:
            self.H_u = 5*self.sigma
        else:
            self.H_u = H_u
            
        if not H_v:
            self.H_v = 5*self.sigma
        else:
            self.H_v = H_v
        
        self._mu_k_arr = None
        self._v_k_arr = None
        self.intervals = defaultdict(list)
        
        #Internal Variables
        self._mu_k = np.array([])
        self._v_k = np.array([])
        self._mu_k_1 = 0
        self._v_k_1 = 0
        self._is_interval = [False]*X.shape[1]
        self._init_idx = [0]*X.shape[1]
        self._final_idx = [0]*X.shape[1]
        self._criteria = None
        
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
    
    def _search_for_change_points(self, idx, criteria):
        """
        Searchs for change points in the filtered data.
        
        Arguments:
            idx: the filtered data sample index
            criteria: the filter to be considered when looking for
                      a change-point (average, variance or both)
        
        Output:
            self._intervals: a list with the initial and final
                             indexes of an interval (if found).
        """
        for col in range(self.X.shape[1]):
            
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
        
    def recursive_exponential_moving_average_and_variance(self, verbose=10, n_jobs=-1):
        """
        Performs a recursive moving average/variance algorithm from past samples
        using a multithread approach.
        
        Arguments:
            verbose: verbose level as in joblib library
            n_jobs: the number of threads as in joblib library
        
        Output:
            self._mu_k_arr: the average filtered data for the given index
            self._v_k_arr: the variance filtered data for the given index
        """
        X = self.X
        results = list(Parallel(n_jobs=n_jobs,
                                require='sharedmem',
                                verbose=verbose)(delayed(self._exponential_moving_average_and_variance)(X, idx)
                                                 for idx in range(len(X))))
         
        self._mu_k_arr, self._v_k_arr = list(zip(*results))
        self._mu_k_arr = np.stack(self._mu_k_arr,axis=0)
        self._v_k_arr = np.stack(self._v_k_arr,axis=0)
        
        self._mu_k_1 = 0
        self._v_k_1 = 0
        
        return self._mu_k_arr, self._v_k_arr
    
    def chenge_points(self, criteria='variance', verbose=10, n_jobs=-1):
        """
        Searchs for change points in the filtered data and its
        corresponding intervals using a multithread approach.
        
        Arguments:
            criteria: the filter to be considered when looking for
                      a change-point (average, variance or both)
            verbose: verbose level as in joblib library
            n_jobs: the number of threads as in joblib library
        """
        #Reset Intervals
        self.intervals = defaultdict(list)
        
        #Update Criteria
        self._criteria = criteria
        
        if (self._mu_k_arr is None) or (self._v_k_arr is None):
            self._mu_k_arr, self._v_k_arr = self.recursive_exponential_moving_average_and_variance(verbose, n_jobs)
            
        Parallel(n_jobs=n_jobs,
                 require='sharedmem',
                 verbose=verbose)(delayed(self._search_for_change_points)(idx,criteria)
                                  for idx in range(len(self.X)))
        
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
        if (self._mu_k_arr is None) or (self._v_k_arr is None):
            self._mu_k_arr, self._v_k_arr = self.recursive_exponential_moving_average_and_variance(verbose, n_jobs)
        
        if not self.intervals:
            intervals = self.chenge_points(verbose=verbose, n_jobs=n_jobs)
        else:
            intervals = self.intervals
        
        for col in list(intervals.keys()):
            intervals_arr = intervals[col]
            
            sns.set_style("darkgrid")
            plt.figure(figsize=(8,5))
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