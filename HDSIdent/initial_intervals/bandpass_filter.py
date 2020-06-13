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
    def __init__(self, X, W, N, sigma = None, H=None, n_jobs=-1, verbose=0):
        """ 
        Constructor.

        Arguments:
            X: the input discrete-time data
            W: input frequency [W1, W2] array as in scipy documentation
            N: Butterworth filter order
            sigma: data (population) standard deviation
            H: change-point threshold
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
        
        if not N:
            self.N = 1
        else:
            self.N = N
    
    def _initialize_internal_variables(self):
        """
        THis function initializes the required
        internal variables.
        """
        self._indicating_sequences = defaultdict(lambda: defaultdict(dict))
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
        
        #Create filtered signal array
        self.butt_mtrx = np.empty(shape=(self.X.shape[0],self.X.shape[1]))
        
        #Define analog filter
        self.num, self.den = signal.butter(N = self.N, 
                                           Wn = self.W, 
                                           btype='bandpass', 
                                           analog=True)
        
        #Compute transfer function
        e = signal.TransferFunction(self.num, self.den)
        
        #Filter each signal (column)
        for col in range(self.X.shape[1]):
            
            #Input initial signal to avoid deflection
            X_aux = [self.X[0,col]]*10000+list(self.X[:,col])
            
            #Filter
            t_in = np.arange(0,len(X_aux),1)
            t_out, butt_arr, _ = signal.lsim(e, X_aux, t_in)
            self.butt_mtrx[:,col] = butt_arr.reshape(-1,1)[10000:,0] 
    
    def _define_deviations_from_the_mean(self):
        """
        Deviation indexes are those in which the absolute
        value of the filtered signal is higher then a given
        threshold H.
        """
        
        #Compute deviations for each signal (column)
        for col in range(self.X.shape[1]):
            indicating_idxs = np.where(np.abs(self.butt_mtrx[:,col]) >= self.H[col])[0]
            self._indicating_sequences[col] = np.zeros(len(self.X[:,col]))
            self._indicating_sequences[col][indicating_idxs] = 1      
    
    def _unify_indicating_sequences(self):
        """
        The resulting indicating sequences are unified to
        obtain a single interval. Let us call the indicating
        sequences for the input Iu and the indicating sequences
        for the output Iy. The unified indicating sequence is
        the defined as Iu U Iy.
        """
        
        indicating_sequence = np.array(self._indicating_sequences[0])
        for key, value in self._indicating_sequences.items():
            if key > 0:
                indicating_sequence = np.maximum(indicating_sequence,
                                                 np.array(value))
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
        
        #Initialize Internal Variables
        self._initialize_internal_variables()
        
        #Apply Butterworth Filter
        self.butterworth_filter()
        
        #Compute Deviations from the mean
        self._define_deviations_from_the_mean()
        
        #Unify Indicating Sequences
        indicating_sequence = self._unify_indicating_sequences()
        
        #Define Intervals
        self.unified_intervals = \
            self._define_intervals_from_indicating_sequence(indicating_sequence=
                                                            indicating_sequence)
        
        return self.unified_intervals
    
    def plot_change_points(self):
        """
        Plots all found change points and its corresponding
        intervals.
        """
        
        #Verify if fit was performed
        if self.unified_intervals is None:
            self.fit()
        
        #Make plot
        for col in range(len(self._indicating_sequences.keys())):
            
            #Take deviation from the mean for current signal
            deviation_idxs = np.argwhere(self._indicating_sequences[col]==1)
            
            #Plot thresholds
            X = self.X
            H = self.H
            
            sns.set_style("darkgrid")
            plt.figure(figsize=(15,5))
            
            #Plot filtered signal
            plt.plot(self.butt_mtrx[:,col], color='coral', linewidth=0.8, zorder=1)
            
            plt.plot([H[col]]*len(X),color='black',linestyle='--')
            plt.annotate(f'+H = {H[col]}',
                         xy=(10*len(X)/10.5, np.around(H[col]+0.1,2)),
                         fontsize=12,
                         fontweight='bold',
                         color='black')
            
            plt.plot([-H[col]]*len(X),color='black',linestyle='--')
            plt.annotate(f'-H = {-H[col]}',
                         xy=(10*len(X)/10.5, np.around(-H[col]-0.1,2)),
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
            
            #Plot deviation from the mean
            plt.scatter(deviation_idxs, 
                        self.butt_mtrx[:,col][deviation_idxs], 
                        s=0.5, 
                        color='darkred',
                        zorder=2)
                    
            plt.show()