import pandas as pd
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

class cusum(object):
    
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
        


    
