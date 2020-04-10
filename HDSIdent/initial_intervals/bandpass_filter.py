import pandas as pd
import numpy as np
from scipy import signal
from collections import defaultdict
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

class BandpassFilter(object):
    """
    Performs signal segmentation using a discrete-time Butterworth
    bandpass filter from SciPy. Notice that the input frequencies are
    normalized into 0 and 1.
    """
    def __init__(self, X, W, N, Ts = None, sigma = None, H=None, n_jobs=-1, verbose=0):
        """ 
        Constructor.

        Arguments:
            X: the input discrete-time data
            W: input frequency [W1, W2] array as in scipy documentation
            N: Butterworth filter order
            sigma: data (population) standard deviation
            H: change-point threshold
            Ts: the sampling frequency of the digital filter (Default = 1.0 seconds)
            n_jobs: the number of CPUs to use
            verbose: the degree of verbosity (going from 0 to 10)
        """
        self.W = W
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
        
        #Internal Variables
        self.unified_intervals = None
        self.intervals = None
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
    
    def _search_for_change_points(self, err_points, idx, col):
        """
        Receives the errors array and use it to find change points and
        its corresponding intervals.
        
        Arguments:
            err_points: an array with error values for camparing with a threshold
            idx: array index
            col: the matrix column (the execution signal)
        """
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
    
            
    def chenge_points(self):
        """
        Computes an error array and use it for finding
        changing points based on a given threshold H.
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

        Parallel(n_jobs=self.n_jobs,
                 require='sharedmem',
                 verbose=self.verbose)(delayed(self._search_for_change_points)(self._err_points,idx,col)
                                       for idx in range(len(X))
                                       for col in range(X.shape[1]))
        
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
    
    def _create_indicating_sequence(self):
        """
        This function creates an indicating sequence, i.e., an array containing 1's
        in the intervals of interest and 0's otherwise, based on each interval obtained
        by the bandpass filter approach.
        
        Output:
            indicating_sequence: the indicating sequence
        """
        indicating_sequence = np.zeros(self.X.shape[0])
        for idx, interval_arr in self.intervals.items():
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
            - Applies the Butterworth Filter in the signal
            - From the filtered signal, defines the initial intervals (change-points)
            - Creates an indicating sequence, unifying input and output intervals
            - From the indicating sequence, creates a final unified interval
        """
        
        #Reset Internal Variables
        self.unified_intervals = None
        self.intervals = None
        self.butt_mtrx = None
        self._is_interval = [False]*self.X.shape[1]
        self._init_idx = [0]*self.X.shape[1]
        self._final_idx = [0]*self.X.shape[1]
        self._err_points = list()
        self._num = 0
        self._den = 0
        
        #Apply Butterworth Filter
        _ = self.butterworth_filter()
        
        #Find change-points
        _ = self.chenge_points()
        
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