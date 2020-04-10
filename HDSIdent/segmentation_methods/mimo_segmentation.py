import numpy as np
import pandas as pd
from scipy.stats import chi2
from collections import defaultdict
from joblib import Parallel, delayed


class MIMOSegmentation(object):
    """
    Performs a Multiple-Input Multiple-Output Segmentation
    for a given model structure and based on the initial
    intervals provided.
    
    This implementation is based on the methods proposed in:

    PERETZKI, D. et al. Data mining of historic data for process identification.
    In: Proceedings of the 2011 AIChE Annual Meeting, p. 1027–1033, 2011.
    
    SHARDT, Y. A. W.; SHAH, S. L. Segmentation Methods for Model Identification from
    Historical Process Data. In: Proceedings of the 19th World Congress.
    Cape Town, South Africa: IFAC, 2014. p. 2836–2841.
    
    BITTENCOURT, A. C. et al. An algorithm for finding process identification
    intervals from normal operating data. Processes, v. 3, p. 357–383, 2015.

    PATEL, A. Data Mining of Process Data in Mutlivariable Systems.
    Degree project in electrical engineering — Royal Institute of Technology,
    Stockholm, Sweden, 2016.
    
    ARENGAS, D.; KROLL, A. A Search Method for Selecting Informative Data in Predominantly
    Stationary Historical Records for Multivariable System Identification.
    In: Proceedings of the 21st International Conference on System Theory,
    Control and Computing (ICSTCC). Sinaia, Romenia: IEEE, 2017a. p. 100–105.
    
    ARENGAS, D.; KROLL, A. Searching for informative intervals in predominantly stationary
    data records to support system identification. In: Proceedings of the XXVI International
    Conference on Information, Communication and Automation Technologies (ICAT). Sarajevo,
    Bosnia-Herzegovina: IEEE, 2017b.
    
    Arguments:
        model_structure: a model structure object defined by the ModelStructure class.
        
        segmentation_method: the segmentation method to be considered, or a list of the
        desired methods. Example: ['method1', 'method2']. The available methods are:
            - 'method1': considers the Condition Number and the Qui-squared test;
            - 'method2': considers the Effective Rank and the Scalar Cross-correlation metric;
            - 'method3': considers the Condition Number and the Scalar Cross-correlation metric;
            - 'method4': considers the Effective Rank and the Qui-Squared test;
            - 'method5': considers the Condition Number, the Qui-Squared Test and the Scalar Cross-correlation metric;
            - 'mehtod6': considers the Effective Rank, the Qui-Squared Test and the Scalar Cross-correlation metric;
            - 'method7': considers the Condition Number, the Effective Rank, the Qui-Squared Test and the Scalar Cross-correlation metric;
 
        parameters_dict: a dictionary with the segmentation parameters. Notice that depending
        on the chosen method, different parameters are required. An example of parameters_dict:
            {'Laguerre':{'qui2_p_value_thr':0.01 <required for methods 1, 4, 5, 6 and 7>, 
                         'cond_thr':300 <required for methods 1, 3, 5 and 7>, 
                         'eff_rank_thr':9 <required for methods 2, 4, 6 and 7>,
                         'cc_thr':3 <required for methods 2, 3, 5, 6 and 7>,
                         'min_input_coupling':1 <always required>,
                         'min_output_coupling':1 <always required>}
             }
        
        segmentation_type: the segmentation type: stationary or incremental. The stationary
        segmentation does not change the initial intervals provided. The incremental segmentation
        augment the initial intervals until the validation conditions are satisfied.
        
        n_jobs: the number of CPUs to use
        verbose: the degree of verbosity (from 0 to 10)
        
    """
    def __init__(self,
                 model_structure,
                 segmentation_method,
                 parameters_dict,
                 segmentation_type='stationary',
                 n_jobs=-1,
                 verbose=0):
        
        self.model_structure = model_structure
        if np.ndim(self.model_structure) == 0:
            self.model_structure = [self.model_structure]
            
        self.segmentation_method = segmentation_method
        if np.ndim(self.segmentation_method) == 0:
            self.segmentation_method = [self.segmentation_method]
            
        self.parameters_dict = parameters_dict
        self.segmentation_type = segmentation_type
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def _initialize_internal_variables(self):
        """
        Initializes class internal variables. The following
        variables are initialized:
        
        self._method_metrics: a dictionary for storing the given methods output metrics.
        self._test_resuts: a dictionary for storing the given methods validation test results.
        self.segments_idx: a dictionary for storing the resulting segments indexes suitable for
        system identification for each method provided.
        """
        
        #Create a Nested Dict
        nested_dict = lambda: defaultdict(nested_dict)
        
        #Initialize Internal Variables
        self._metrics_dict = defaultdict(nested_dict)
        self._test_resuts = defaultdict(nested_dict)
        self.segments_idx = defaultdict(nested_dict)
        self._segment_sucesses_dict = defaultdict(nested_dict)
        self.sucessed_intervals = defaultdict(nested_dict)
        self.tests_results = defaultdict(nested_dict)
    
    def _compute_model_metrics(self, X, y):
        """
        This function takes the metrics computed by
        a particular model structure and stores them
        in an internal variable _metrics_dict. The
        following metrics are extracted:
        
        - miso_ranks: the effective ranks
        - miso_correlations: the scalar cross-correlation
        - cond_num_dict: the condition numbers
        - qui_squared_dict: the qui-squared test results
        
        Arguments:
            X: the input signal matrix. Each column corresponds
            to a unique signal;
            
            y: the output signal matrix. Each column corresponds
            to a unique signal.
        """
        for structure in self.model_structure:
            
            if self.verbose > 0:
                print(f"Fitting {structure.name} Model Structure...")
                
            (miso_ranks, 
             miso_correlations, 
             cond_num_dict,
             qui_squared_dict) = structure.fit(X=X,y=y)    

            self._metrics_dict[structure.name]['miso_ranks'] = miso_ranks
            self._metrics_dict[structure.name]['miso_correlations'] = miso_correlations
            self._metrics_dict[structure.name]['cond_num_dict'] = cond_num_dict
            self._metrics_dict[structure.name]['qui_squared_dict'] = qui_squared_dict
            
            if self.verbose > 0:
                print(f"{structure.name} Structure fit finished!")
                
    def _method1(self, method, structure, interval_idx):
        """
        This segmentation method considers an interval suitable
        for System Identification if the following tests are 
        satisfied:
        
        1) The Condition Number of a given interval is lower then
        its provided threshold;
        
        2) The qui-squared computed statistic is greater than the
        critical value for a given p-value.
        
        This test if performed for every combination of input/output
        for every MISO system in the MIMO data provided. An interval
        will only be considered if the number of successes satisfy the
        coupling condition. For a 2x2 system, for example, if the system 
        is decoupled, both inputs must satisfy the criteria above for each
        MISO system. In this case, we must have 4 sucesses: 2 for each
        MISO system.
        
        Arguments:
            method: the method name;
            structure: the model structure being considered (Ex: Laguerre, AR and ARX);
            interval_idx: the interval index being considered in the iteration.
        """
        #Iterate over each column/input (miso system)
        num_output_sucesses = 0
        for output_idx in structure.cond_num_dict['segment_'+str(interval_idx)].keys():
            num_input_sucesses = 0
            input_sucesses = []
            for input_idx in structure.cond_num_dict['segment_'+str(interval_idx)][output_idx].keys():
                #Method 1 Rules
                cond_num = self._metrics_dict[structure.name]\
                                             ['cond_num_dict']\
                                             ['segment_'+str(interval_idx)]\
                                             [output_idx]\
                                             [input_idx]

                qui_squared = self._metrics_dict[structure.name]\
                                                ['qui_squared_dict']\
                                                ['segment_'+str(interval_idx)]\
                                                [output_idx]\
                                                [input_idx]

                qui_thr = chi2.ppf(1-self.parameters_dict[structure.name]['qui2_p_value_thr'],structure.Nb)
                cond_thr = self.parameters_dict[structure.name]['cond_thr']

                if ((cond_num <= cond_thr) and (qui_squared >= qui_thr)):
                    num_input_sucesses += 1
                    input_sucesses.append(input_idx)
                    
                    #Store Input/Output Segment that succeeded
                    self._segment_sucesses_dict[method]\
                                               [structure.name]\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx] = input_sucesses
                    
                    #Store test result (success or fail)
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = True
                else:
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = False

            if num_input_sucesses >= self.parameters_dict[structure.name]['min_input_coupling']:
                num_output_sucesses += 1
        
        #Consider the interval valid only if the desired number of inputs and outputs succeeded
        if num_output_sucesses >= self.parameters_dict[structure.name]['min_output_coupling']:
            self.sucessed_intervals[method]\
                                   [structure.name]\
                                   ['segment_'+str(interval_idx)] = \
                structure.initial_intervals[interval_idx]
    
    def _method2(self, method, structure, interval_idx):
        """
        This segmentation method considers an interval suitable
        for System Identification if the following tests are 
        satisfied:
        
        1) The Effective Rank of a given interval is higher then
        its provided threshold;
        
        2) The scalar cross-correlation metric is greater than the
        its provided threshold.
        
        This test if performed for every combination of input/output
        for every MISO system in the MIMO data provided. An interval
        will only be considered if the number of successes satisfy the
        coupling condition. For a 2x2 system, for example, if the system 
        is decoupled, both inputs must satisfy the criteria above for each
        MISO system. In this case, we must have 4 sucesses: 2 for each
        MISO system.
        
        Arguments:
            method: the method name;
            structure: the model structure being considered (Ex: Laguerre, AR and ARX);
            interval_idx: the interval index being considered in the iteration.
        """
        #Iterate over each column/input (miso system)
        num_output_sucesses = 0
        for output_idx in structure.cond_num_dict['segment_'+str(interval_idx)].keys():
            num_input_sucesses = 0
            input_sucesses = []
            for input_idx in structure.cond_num_dict['segment_'+str(interval_idx)][output_idx].keys():
                #Method 2 Rules
                eff_rank = self._metrics_dict[structure.name]\
                                             ['miso_ranks']\
                                             ['segment_'+str(interval_idx)]\
                                             [output_idx]\
                                             [input_idx]

                cross_corr = self._metrics_dict[structure.name]\
                                               ['miso_correlations']\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx]\
                                               [input_idx]

                eff_rank_thr = self.parameters_dict[structure.name]['eff_rank_thr']
                cc_thr = self.parameters_dict[structure.name]['cc_thr']

                if ((eff_rank >= eff_rank_thr) and (cross_corr >= cc_thr)):
                    num_input_sucesses += 1
                    input_sucesses.append(input_idx)
                    
                    #Store Input/Output Segment that succeeded
                    self._segment_sucesses_dict[method]\
                                               [structure.name]\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx] = input_sucesses
                    
                    #Store test result (success or fail)
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = True
                else:
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = False

            if num_input_sucesses >= self.parameters_dict[structure.name]['min_input_coupling']:
                num_output_sucesses += 1
        
        #Consider the interval valid only if the desired number of inputs and outputs succeeded
        if num_output_sucesses >= self.parameters_dict[structure.name]['min_output_coupling']:
            self.sucessed_intervals[method]\
                                   [structure.name]\
                                   ['segment_'+str(interval_idx)] = \
                structure.initial_intervals[interval_idx]
    
    def _method3(self, method, structure, interval_idx):
        """
        This segmentation method considers an interval suitable
        for System Identification if the following tests are 
        satisfied:
        
        1) The Condition Number of a given interval is lower then
        its provided threshold;
        
        2) The scalar cross-correlation metric is greater than the
        its provided threshold.
        
        This test if performed for every combination of input/output
        for every MISO system in the MIMO data provided. An interval
        will only be considered if the number of successes satisfy the
        coupling condition. For a 2x2 system, for example, if the system 
        is decoupled, both inputs must satisfy the criteria above for each
        MISO system. In this case, we must have 4 sucesses: 2 for each
        MISO system.
        
        Arguments:
            method: the method name;
            structure: the model structure being considered (Ex: Laguerre, AR and ARX);
            interval_idx: the interval index being considered in the iteration.
        """
        #Iterate over each column/input (miso system)
        num_output_sucesses = 0
        for output_idx in structure.cond_num_dict['segment_'+str(interval_idx)].keys():
            num_input_sucesses = 0
            input_sucesses = []
            for input_idx in structure.cond_num_dict['segment_'+str(interval_idx)][output_idx].keys():
                #Method 3 Rules
                cond_num = self._metrics_dict[structure.name]\
                                             ['cond_num_dict']\
                                             ['segment_'+str(interval_idx)]\
                                             [output_idx]\
                                             [input_idx]
                
                cross_corr = self._metrics_dict[structure.name]\
                                               ['miso_correlations']\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx]\
                                               [input_idx]

                cond_thr = self.parameters_dict[structure.name]['cond_thr']
                cc_thr = self.parameters_dict[structure.name]['cc_thr']

                if ((cond_num <= cond_thr) and (cross_corr >= cc_thr)):
                    num_input_sucesses += 1
                    input_sucesses.append(input_idx)
                    
                    #Store Input/Output Segment that succeeded
                    self._segment_sucesses_dict[method]\
                                               [structure.name]\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx] = input_sucesses
                    
                    #Store test result (success or fail)
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = True
                else:
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = False

            if num_input_sucesses >= self.parameters_dict[structure.name]['min_input_coupling']:
                num_output_sucesses += 1
        
        #Consider the interval valid only if the desired number of inputs and outputs succeeded
        if num_output_sucesses >= self.parameters_dict[structure.name]['min_output_coupling']:
            self.sucessed_intervals[method]\
                                   [structure.name]\
                                   ['segment_'+str(interval_idx)] = \
                structure.initial_intervals[interval_idx]
            
    def _method4(self, method, structure, interval_idx):
        """
        This segmentation method considers an interval suitable
        for System Identification if the following tests are 
        satisfied:
        
        1) The Effective Rank of a given interval is higher then
        its provided threshold;
        
        2) The qui-squared computed statistic is greater than the
        critical value for a given p-value.
        
        This test if performed for every combination of input/output
        for every MISO system in the MIMO data provided. An interval
        will only be considered if the number of successes satisfy the
        coupling condition. For a 2x2 system, for example, if the system 
        is decoupled, both inputs must satisfy the criteria above for each
        MISO system. In this case, we must have 4 sucesses: 2 for each
        MISO system.
        
        Arguments:
            method: the method name;
            structure: the model structure being considered (Ex: Laguerre, AR and ARX);
            interval_idx: the interval index being considered in the iteration.
        """
        #Iterate over each column/input (miso system)
        num_output_sucesses = 0
        for output_idx in structure.cond_num_dict['segment_'+str(interval_idx)].keys():
            num_input_sucesses = 0
            input_sucesses = []
            for input_idx in structure.cond_num_dict['segment_'+str(interval_idx)][output_idx].keys():
                #Method 4 Rules
                eff_rank = self._metrics_dict[structure.name]\
                                             ['miso_ranks']\
                                             ['segment_'+str(interval_idx)]\
                                             [output_idx]\
                                             [input_idx]
                
                qui_squared = self._metrics_dict[structure.name]\
                                                ['qui_squared_dict']\
                                                ['segment_'+str(interval_idx)]\
                                                [output_idx]\
                                                [input_idx]

                eff_rank_thr = self.parameters_dict[structure.name]['eff_rank_thr']
                qui_thr = chi2.ppf(1-self.parameters_dict[structure.name]['qui2_p_value_thr'],structure.Nb)

                if ((eff_rank >= eff_rank_thr) and (qui_squared >= qui_thr)):
                    num_input_sucesses += 1
                    input_sucesses.append(input_idx)
                    
                    #Store Input/Output Segment that succeeded
                    self._segment_sucesses_dict[method]\
                                               [structure.name]\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx] = input_sucesses
                    
                    #Store test result (success or fail)
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = True
                else:
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = False

            if num_input_sucesses >= self.parameters_dict[structure.name]['min_input_coupling']:
                num_output_sucesses += 1
        
        #Consider the interval valid only if the desired number of inputs and outputs succeeded
        if num_output_sucesses >= self.parameters_dict[structure.name]['min_output_coupling']:
            self.sucessed_intervals[method]\
                                   [structure.name]\
                                   ['segment_'+str(interval_idx)] = \
                structure.initial_intervals[interval_idx]
            
    def _method5(self, method, structure, interval_idx):
        """
        This segmentation method considers an interval suitable
        for System Identification if the following tests are 
        satisfied:
        
        1) The Condition Number of a given interval is lower then
        its provided threshold;
        
        2) The qui-squared computed statistic is greater than the
        critical value for a given p-value;
        
        3) The scalar cross-correlation metric is greater than the
        its provided threshold.
        
        This test if performed for every combination of input/output
        for every MISO system in the MIMO data provided. An interval
        will only be considered if the number of successes satisfy the
        coupling condition. For a 2x2 system, for example, if the system 
        is decoupled, both inputs must satisfy the criteria above for each
        MISO system. In this case, we must have 4 sucesses: 2 for each
        MISO system.
        
        Arguments:
            method: the method name;
            structure: the model structure being considered (Ex: Laguerre, AR and ARX);
            interval_idx: the interval index being considered in the iteration.
        """
        #Iterate over each column/input (miso system)
        num_output_sucesses = 0
        for output_idx in structure.cond_num_dict['segment_'+str(interval_idx)].keys():
            num_input_sucesses = 0
            input_sucesses = []
            for input_idx in structure.cond_num_dict['segment_'+str(interval_idx)][output_idx].keys():
                #Method 5 Rules
                cond_num = self._metrics_dict[structure.name]\
                                             ['cond_num_dict']\
                                             ['segment_'+str(interval_idx)]\
                                             [output_idx]\
                                             [input_idx]

                qui_squared = self._metrics_dict[structure.name]\
                                                ['qui_squared_dict']\
                                                ['segment_'+str(interval_idx)]\
                                                [output_idx]\
                                                [input_idx]

                cross_corr = self._metrics_dict[structure.name]\
                                               ['miso_correlations']\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx]\
                                               [input_idx]
                
                qui_thr = chi2.ppf(1-self.parameters_dict[structure.name]['qui2_p_value_thr'],structure.Nb)
                cond_thr = self.parameters_dict[structure.name]['cond_thr']
                cc_thr = self.parameters_dict[structure.name]['cc_thr']
                
                if ((cond_num <= cond_thr) and (qui_squared >= qui_thr) and (cross_corr >= cc_thr)):
                    num_input_sucesses += 1
                    input_sucesses.append(input_idx)
                    
                    #Store Input/Output Segment that succeeded
                    self._segment_sucesses_dict[method]\
                                               [structure.name]\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx] = input_sucesses
                    
                    #Store test result (success or fail)
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = True
                else:
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = False

            if num_input_sucesses >= self.parameters_dict[structure.name]['min_input_coupling']:
                num_output_sucesses += 1
        
        #Consider the interval valid only if the desired number of inputs and outputs succeeded
        if num_output_sucesses >= self.parameters_dict[structure.name]['min_output_coupling']:
            self.sucessed_intervals[method]\
                                   [structure.name]\
                                   ['segment_'+str(interval_idx)] = \
                structure.initial_intervals[interval_idx]
            
    def _method6(self, method, structure, interval_idx):
        """
        This segmentation method considers an interval suitable
        for System Identification if the following tests are 
        satisfied:
        
        1) The Effective Rank of a given interval is higher then
        its provided threshold;
        
        2) The qui-squared computed statistic is greater than the
        critical value for a given p-value;
        
        3) The scalar cross-correlation metric is greater than the
        its provided threshold.
        
        This test if performed for every combination of input/output
        for every MISO system in the MIMO data provided. An interval
        will only be considered if the number of successes satisfy the
        coupling condition. For a 2x2 system, for example, if the system 
        is decoupled, both inputs must satisfy the criteria above for each
        MISO system. In this case, we must have 4 sucesses: 2 for each
        MISO system.
        
        Arguments:
            method: the method name;
            structure: the model structure being considered (Ex: Laguerre, AR and ARX);
            interval_idx: the interval index being considered in the iteration.
        """
        #Iterate over each column/input (miso system)
        num_output_sucesses = 0
        for output_idx in structure.cond_num_dict['segment_'+str(interval_idx)].keys():
            num_input_sucesses = 0
            input_sucesses = []
            for input_idx in structure.cond_num_dict['segment_'+str(interval_idx)][output_idx].keys():
                #Method 6 Rules
                eff_rank = self._metrics_dict[structure.name]\
                                             ['miso_ranks']\
                                             ['segment_'+str(interval_idx)]\
                                             [output_idx]\
                                             [input_idx]

                qui_squared = self._metrics_dict[structure.name]\
                                                ['qui_squared_dict']\
                                                ['segment_'+str(interval_idx)]\
                                                [output_idx]\
                                                [input_idx]

                cross_corr = self._metrics_dict[structure.name]\
                                               ['miso_correlations']\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx]\
                                               [input_idx]
                
                eff_rank_thr = self.parameters_dict[structure.name]['eff_rank_thr']
                qui_thr = chi2.ppf(1-self.parameters_dict[structure.name]['qui2_p_value_thr'],structure.Nb)
                cc_thr = self.parameters_dict[structure.name]['cc_thr']
                
                if ((eff_rank >= eff_rank_thr) and (qui_squared >= qui_thr) and (cross_corr >= cc_thr)):
                    num_input_sucesses += 1
                    input_sucesses.append(input_idx)
                    
                    #Store Input/Output Segment that succeeded
                    self._segment_sucesses_dict[method]\
                                               [structure.name]\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx] = input_sucesses
                    
                    #Store test result (success or fail)
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = True
                else:
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = False

            if num_input_sucesses >= self.parameters_dict[structure.name]['min_input_coupling']:
                num_output_sucesses += 1
        
        #Consider the interval valid only if the desired number of inputs and outputs succeeded
        if num_output_sucesses >= self.parameters_dict[structure.name]['min_output_coupling']:
            self.sucessed_intervals[method]\
                                   [structure.name]\
                                   ['segment_'+str(interval_idx)] = \
                structure.initial_intervals[interval_idx]
            
    def _method7(self, method, structure, interval_idx):
        """
        This segmentation method considers an interval suitable
        for System Identification if the following tests are 
        satisfied:
        
        1) The Condition Number of a given interval is lower then
        its provided threshold;
        
        2) The Effective Rank of a given interval is higher then
        its provided threshold;
        
        3) The qui-squared computed statistic is greater than the
        critical value for a given p-value;
        
        4) The scalar cross-correlation metric is greater than the
        its provided threshold.
        
        This test if performed for every combination of input/output
        for every MISO system in the MIMO data provided. An interval
        will only be considered if the number of successes satisfy the
        coupling condition. For a 2x2 system, for example, if the system 
        is decoupled, both inputs must satisfy the criteria above for each
        MISO system. In this case, we must have 4 sucesses: 2 for each
        MISO system.
        
        Arguments:
            method: the method name;
            structure: the model structure being considered (Ex: Laguerre, AR and ARX);
            interval_idx: the interval index being considered in the iteration.
        """
        #Iterate over each column/input (miso system)
        num_output_sucesses = 0
        for output_idx in structure.cond_num_dict['segment_'+str(interval_idx)].keys():
            num_input_sucesses = 0
            input_sucesses = []
            for input_idx in structure.cond_num_dict['segment_'+str(interval_idx)][output_idx].keys():
                #Method 7 Rules
                cond_num = self._metrics_dict[structure.name]\
                                             ['cond_num_dict']\
                                             ['segment_'+str(interval_idx)]\
                                             [output_idx]\
                                             [input_idx]
                
                eff_rank = self._metrics_dict[structure.name]\
                                             ['miso_ranks']\
                                             ['segment_'+str(interval_idx)]\
                                             [output_idx]\
                                             [input_idx]

                qui_squared = self._metrics_dict[structure.name]\
                                                ['qui_squared_dict']\
                                                ['segment_'+str(interval_idx)]\
                                                [output_idx]\
                                                [input_idx]

                cross_corr = self._metrics_dict[structure.name]\
                                               ['miso_correlations']\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx]\
                                               [input_idx]
                
                cond_thr = self.parameters_dict[structure.name]['cond_thr']
                eff_rank_thr = self.parameters_dict[structure.name]['eff_rank_thr']
                qui_thr = chi2.ppf(1-self.parameters_dict[structure.name]['qui2_p_value_thr'],structure.Nb)
                cc_thr = self.parameters_dict[structure.name]['cc_thr']
                
                if ((eff_rank >= eff_rank_thr) and (cond_num <= cond_thr) and 
                    (qui_squared >= qui_thr) and (cross_corr >= cc_thr)):
                    num_input_sucesses += 1
                    input_sucesses.append(input_idx)
                    
                    #Store Input/Output Segment that succeeded
                    self._segment_sucesses_dict[method]\
                                               [structure.name]\
                                               ['segment_'+str(interval_idx)]\
                                               [output_idx] = input_sucesses
                    
                    #Store test result (success or fail)
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = True
                else:
                    self.tests_results[method]\
                                      [structure.name]\
                                      ['segment_'+str(interval_idx)]\
                                      [output_idx]\
                                      [input_idx] = False

            if num_input_sucesses >= self.parameters_dict[structure.name]['min_input_coupling']:
                num_output_sucesses += 1
        
        #Consider the interval valid only if the desired number of inputs and outputs succeeded
        if num_output_sucesses >= self.parameters_dict[structure.name]['min_output_coupling']:
            self.sucessed_intervals[method]\
                                   [structure.name]\
                                   ['segment_'+str(interval_idx)] = \
                structure.initial_intervals[interval_idx]
            
    def _apply_stationary_segmentation(self):
        """
        This function applies a stationary MIMO segmentation
        for System Identification. The stationary segmentation
        considers the initial intervals provided as they are, i.e.,
        the provided intervals are tested againts the segmentation
        hypothesis and, in case they succeed, they are considered
        suitable for System Identification, otherwise they are 
        discarded. This method do not modify the original intervals
        in any fashion.
        """
        
        for method in self.segmentation_method:
            
            #Make an Parallel executor
            executor = Parallel(require='sharedmem',
                                n_jobs=self.n_jobs,
                                verbose=self.verbose)
            
            #Make Segmentation
            method_func = getattr(self, '_' + method)
            
            if self.verbose > 0:
                print(f"Beginning Stationary Segmentation for {method}...")
                
            method_task = (delayed(method_func)(method, structure, interval_idx)
                           for structure in self.model_structure
                           for interval_idx in structure.initial_intervals.keys())
            executor(method_task)  
            
            if self.verbose > 0:
                print("Stationary Segmentation Finished!")
                
    def fit(self, X, y):
        """
        This function performs all the steps required for
        performing a MIMO Segmentation or System Identification
        based on historical data X and y.
        
        Arguments:
            X: the input signal matrix. Each column corresponds
            to a unique signal;
            
            y: the output signal matrix. Each column corresponds
            to a unique signal.
        """
        
        #Initialize Internal Variables
        self._initialize_internal_variables()
        
        #Fit model Structures
        self._compute_model_metrics(X=X,y=y)
        
        #Make Segmentation
        if self.segmentation_type == 'stationary':
            self._apply_stationary_segmentation()