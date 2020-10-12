from scipy.stats import norm as sci_norm
from collections import defaultdict
from numpy import linalg as LA
import pandas as pd
import numpy as np


class ModelStructure(object):
    """
    This class contains general functions that can be
    used for a variety of model structures used in
    System Identification. This class is used as a
    father class for model structure classes.

    Reference works:
        PERETZKI, D. et al. Data mining of historic data for process identification.
        In: Proceedings of the 2011 AIChE Annual Meeting, p. 1027–1033, 2011.

        SHARDT, Y. A. W.; SHAH, S. L. Segmentation Methods for Model Identification from
        Historical Process Data. In: Proceedings of the 19th World Congress.
        Cape Town, South Africa: IFAC, 2014. p. 2836–2841.

        AGUIRRE, L. A. Introdução à Identificação de Sistemas:
        técnicas lineares e não lineares: teoria e aplicação. 4. ed.
        Belo Horizonte, Brasil: Editora UFMG, 2015.

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
    """

    def __init__(self):
        pass

    def _verify_data(self, X, y):
        """
        Verifies the data type and save data columns
        in case they are provided.

        Arguments:
            X: the input data in pandas dataframe format or numpy array
            y: the output data in pandas dataframe format or numpy array

        Output:
            X: the input data in numpy array format
            y: the input data in numpy array format
            X_cols: the input data columns in case they are provided
            y_cols: the output data columns in case they are provided
        """
        if type(X) == pd.core.frame.DataFrame:
            X_cols = X.columns
            X = X.values
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        elif type(X) == np.ndarray:
            X_cols = None
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        else:
            raise Exception("Input data must be a pandas dataframe or a numpy array")

        if type(y) == pd.core.frame.DataFrame:
            y_cols = y.columns
            y = y.values
            if y.ndim == 1:
                y = y.reshape(-1, 1)
        elif type(y) == np.ndarray:
            y_cols = None
            if y.ndim == 1:
                y = y.reshape(-1, 1)
        else:
            raise Exception("Input data must be a pandas dataframe or a numpy array")

        return X, y, X_cols, y_cols

    def _initialize_metrics(self, X, y, X_cols, y_cols):
        """
        This function initializes the following metrics:
            - Phi_dict: a dictionary of regressor matrices for each input/output and for each signal;
            - I_dict: a dictionary of information matrices of the form [Phi]^T[Phi];
            - cond_num_dict: a dictionary of condition numbers for each information matrix;
            - theta_dict: a dictionary of estimated parameter vectors phi = [ph1 ph2 ... phiNb];
            - chi_squared_dict: a dictionary of chi-squared test for validating the estimated parameters;
            - cross_corr_dict: a dictionary of cross-correlations for each input/output;
            - eff_rank_1_dict: a dictionary of type 1 effective ranks;
            - eff_rank_2_dict: a dictionary of type 2 effective ranks;
            - miso_ranks: a dictionary of effective ranks;
            - miso_correlations: a dictionary of cross-corellations for each input/output and for each signal;
            - Phi_aug_dict: a dictionary of augmented matrices of the form [Phi y].
        """
        # Create Internal Variables
        self.Phi_dict = defaultdict(lambda: defaultdict(dict))
        self.I_dict = defaultdict(lambda: defaultdict(dict))
        self.cond_num_dict = defaultdict(lambda: defaultdict(dict))
        self.theta_dict = defaultdict(lambda: defaultdict(dict))
        self.chi_squared_dict = defaultdict(lambda: defaultdict(dict))
        self.cross_corr_dict = defaultdict(lambda: defaultdict(dict))
        self.eff_rank_1_dict = defaultdict(lambda: defaultdict(dict))
        self.eff_rank_2_dict = defaultdict(lambda: defaultdict(dict))
        self.miso_ranks = defaultdict(lambda: defaultdict(dict))
        self.miso_correlations = defaultdict(lambda: defaultdict(dict))
        self.Phi_aug_dict = defaultdict(lambda: defaultdict(dict))

    def _update_index_name(self, input_idx, X_cols, output_idx, y_cols):
        """
        This function verifies if the provided data contains
        column names. In the case it does, the column name is
        used as index, otherwise the index number is concatenated
        with the word input or output, depending on the signal type.
        """
        if X_cols is not None:
            input_idx_name = X_cols[input_idx]
        else:
            input_idx_name = "input" + "_" + str(input_idx)

        if y_cols is not None:
            output_idx_name = y_cols[output_idx]
        else:
            output_idx_name = "output" + "_" + str(output_idx)

        return input_idx_name, output_idx_name

    def _qr_factorization(
        self, y, input_idx, X_cols, output_idx, y_cols, segment, operation
    ):
        """
        Performs a QR-Factorization (Decomposition) using numpy linear
        algebra library and uses the R matrix to solve the Ordinary Least
        Square (OLS) problem.

        Arguments:
            y: the ouput signals
            input_idx: the sequential number of the execution input;
            X_cols: the input data columns in case they are provided;
            output_idx: the sequential number of the execution output;
            y_cols: the output data columns in case they are provided;
            segment: the sequential number of the execution segment (interval).
            operation: which operation to perform (all, condition_number or chi_squared_test)
        """

        # Take Column Names
        input_idx_name, output_idx_name = self._update_index_name(
            input_idx, X_cols, output_idx, y_cols
        )

        # Take Segment
        segment_idx = self.initial_intervals[segment]

        # Take Regressor Matrix
        Phi = self.Phi_dict["segment" + "_" + str(segment)][output_idx_name][
            input_idx_name
        ]

        # Define the y shift according to the model structure
        # If a model structure is of order 3, for example, the
        # output used for fitting the model must start 3 samples
        # ahead. In that case, y_shift=3. For Laguerre models, the
        # y_shift is always 1, regardless of the model order.
        y_length = len(y[segment_idx, output_idx])
        regressor_length = Phi.shape[0]
        y_shift = y_length - regressor_length

        # Create the Augmented Regressor Matrix [Phi y]
        self.Phi_aug_dict["segment" + "_" + str(segment)][output_idx_name][
            input_idx_name
        ] = np.zeros((len(segment_idx[y_shift:]), self.Nb + 1))

        self.Phi_aug_dict["segment" + "_" + str(segment)][output_idx_name][
            input_idx_name
        ][: Phi.shape[0], : self.Nb] = Phi

        self.Phi_aug_dict["segment" + "_" + str(segment)][output_idx_name][
            input_idx_name
        ][:, -1] = np.squeeze(y[segment_idx, output_idx][y_shift:])

        # QR-Factorization
        Q, R = LA.qr(
            self.Phi_aug_dict["segment" + "_" + str(segment)][output_idx_name][
                input_idx_name
            ]
        )
        R1 = R[: self.Nb, : self.Nb]
        R2 = R[: self.Nb, self.Nb]
        R3 = R[self.Nb, self.Nb]

        # Comput Theta, Information Matrix and its Condition Number and the chi-squared Test
        if operation in ("all", "condition_number"):
            self.I_dict["segment" + "_" + str(segment)][output_idx_name][
                input_idx_name
            ] = (1 / len(np.squeeze(y[segment_idx, output_idx][y_shift:]))) * np.matmul(
                R1.T, R1
            )

            self.cond_num_dict["segment" + "_" + str(segment)][output_idx_name][
                input_idx_name
            ] = LA.cond(
                self.I_dict["segment" + "_" + str(segment)][output_idx_name][
                    input_idx_name
                ]
            )

        if operation in ("all", "chi_squared_test"):
            try:
                self.theta_dict["segment" + "_" + str(segment)][output_idx_name][
                    input_idx_name
                ] = np.matmul(LA.inv(R1), R2)
            except:
                pass

            self.chi_squared_dict["segment" + "_" + str(segment)][output_idx_name][
                input_idx_name
            ] = (
                np.sqrt(len(np.squeeze(y[segment_idx, output_idx][y_shift:])))
                / np.abs(R3)
            ) * LA.norm(
                x=R2, ord=2
            )

    def _cross_correlation_scalar_metric(self, X, y, delay, cc_alpha):
        """
        Computes a scalar metric that represents the cross-correlation
        function for a range of lag values. The lag range goes from
        -delay to delay. The normalized cross-correlation is computed
        for signals X and y and compared to the critical value of a
        two-sided normal distribution for an alpha confidence value.

        This metric is proposed in the following reference:

        RIBEIRO, A. H.; AGUIRRE, L. A. Selecting transients automatically
        for the identification of models for an oil well. IFAC-PapersOnLine,
        v. 48, n. 6, p. 154–158, 2015.

        Arguments:
            X: the input signal;
            y: the output signal;
            delay: the maximum/minimum cross-correlation lag value between the input and the output signals;
            cc_alpha: the confidence value for a normal distribution.

        Output:
            ccsm: the cross-correlation scalar metric.
        """
        # Compute p-value
        p = sci_norm.ppf(1 - (cc_alpha) / 2) / np.sqrt(len(X))

        s_arr = []
        for d in range(-delay, delay + 1):
            # Compute Normalized Cross Corellation for current delay
            ncc = self._normalized_cross_correlation(X=X, y=y, delay=d)

            if np.abs(ncc) <= p:
                s_arr.append(0)
            elif np.abs(ncc) > p and d != 0:
                s_arr.append((np.abs(ncc) - p) / np.abs(d))
            else:
                s_arr.append(np.abs(ncc) - p)

        ccsm = np.sum(s_arr)

        return ccsm

    def _normalized_cross_correlation(self, X, y, delay):
        """
        Computes the normalized cross-correlation function
        of signals X and y for a given delay value.

        Arguments:
            X: the input signal;
            y: the output signal;
            delay: the delay between both signals.

        Output:
            ncc: the normalized cross-correlation value.
        """
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        if delay < 0:
            num = np.sum(
                [
                    (X[idx] - X_mean) * (y[idx + delay] - y_mean)
                    for idx in range(np.abs(delay), len(X))
                ]
            )
            den_1 = np.sum(
                [(X[idx] - X_mean) ** 2 for idx in range(np.abs(delay), len(X))]
            )
            den_2 = np.sum(
                [(y[idx + delay] - y_mean) ** 2 for idx in range(np.abs(delay), len(X))]
            )
            den = np.sqrt(den_1 * den_2)
        else:
            num = np.sum(
                [
                    (X[idx] - X_mean) * (y[idx + delay] - y_mean)
                    for idx in range(0, len(X) - delay)
                ]
            )
            den_1 = np.sum([(X[idx] - X_mean) ** 2 for idx in range(0, len(X) - delay)])
            den_2 = np.sum(
                [(y[idx + delay] - y_mean) ** 2 for idx in range(0, len(X) - delay)]
            )
            den = np.sqrt(den_1 * den_2)

        if den == 0:
            ncc = 0
        else:
            ncc = num / den

        return ncc

    def _effective_rank_type_2(self, singular_values, threshold):
        """
        Compute the effective rank as a function of the difference
        of two consecutive singular values.

        This implementation was based on the following reference:

        RIBEIRO, A. H.; AGUIRRE, L. A. Selecting transients automatically
        for the identification of models for an oil well. IFAC-PapersOnLine,
        v. 48, n. 6, p. 154–158, 2015.

        Arguments:
            singular_values: matrix singular values;
            threshold: effective rank threshold.

        Output:
            efr: the computed effective rank.
        """

        efr_arr = []
        for idx in range(1, len(singular_values)):
            # Compute Consecutives Singular Values
            s_i_1 = singular_values[idx - 1]
            s_i = singular_values[idx]

            # Compute the difference of the consecutive singular values
            s_diff = s_i_1 - s_i

            # Compute effective rank for index idx
            if s_diff > threshold:
                efr_arr.append(1)
            else:
                efr_arr.append(0)

        efr = np.sum(efr_arr)

        return efr

    def _effective_rank_type_1(self, singular_values, threshold):
        """
        Compute the effective rank as a function of the normalized
        singular values.

        This implementation was based on the following reference:

        RIBEIRO, A. H.; AGUIRRE, L. A. Selecting transients automatically
        for the identification of models for an oil well. IFAC-PapersOnLine,
        v. 48, n. 6, p. 154–158, 2015.

        Arguments:
            singular_values: matrix singular values;
            threshold: effective rank threshold.

        Output:
            efr: the computed effective rank.
        """

        # Compute L1-norm
        l1_norm = np.sum([np.abs(s) for s in singular_values])

        # Compute Normalized Singular Values
        p_arr = [s / l1_norm for s in singular_values]

        # Compute Effective Rank for given Threshold
        efr = np.sum([1 if p > threshold else 0 for p in p_arr])

        return efr

    def _effective_rank(self, A, threshold, efr_type):
        """
        Compute the effective rank of matrix A for
        a given threshold. Two types of effective
        rank are available and implemented based on the
        following reference:

        RIBEIRO, A. H.; AGUIRRE, L. A. Selecting transients automatically
        for the identification of models for an oil well. IFAC-PapersOnLine,
        v. 48, n. 6, p. 154–158, 2015.

        Arguments:
            A: the input matrix;
            threshold: the threshold for computing the effective rank;
            efr_type: type_1 or type_2.

        Output:
            efr: the effective rank
        """

        # Compute Singular Values of Matrix A
        _, singular_values, _ = LA.svd(A)

        # Compute Effective Rank
        if efr_type == "type_1":
            return self._effective_rank_type_1(
                singular_values=singular_values, threshold=threshold
            )
        elif efr_type == "type_2":
            return self._effective_rank_type_2(
                singular_values=singular_values, threshold=threshold
            )

    def _compute_miso_ranks(
        self, X, y, regressor_mtrx, input_idx, X_cols, output_idx, y_cols, segment
    ):
        """
        For each MISO System, i.e., for each output, compute the effective rank
        of the AR Information matrix for the corresponding output.

        Arguments:
            y: the output signal
            output_idx: the sequential number of the execution output
            segment: the sequential number of the execution segment (interval)
        """
        # Take Column Names
        input_idx_name, output_idx_name = self._update_index_name(
            input_idx, X_cols, output_idx, y_cols
        )

        # Compute the Effective Rank of the Information Matrix
        efr = self._effective_rank(
            A=regressor_mtrx, threshold=self.sv_thr, efr_type=self.efr_type
        )

        self.miso_ranks["segment" + "_" + str(segment)][output_idx_name][
            input_idx_name
        ] = efr

    def _compute_miso_correlations(
        self, X, y, input_idx, X_cols, output_idx, y_cols, segment
    ):
        """
        For each MISO System, i.e., for each output, compute the cross-correlation
        metric between each input and the corresponding output.

        Arguments:
            X: the input signal
            y: the output signal
            input_idx: the sequential number of the execution input
            output_idx: the sequential number of the execution output
            segment: the sequential number of the execution segment (interval)
        """
        # Take Column Names
        input_idx_name, output_idx_name = self._update_index_name(
            input_idx, X_cols, output_idx, y_cols
        )

        ncc = self._cross_correlation_scalar_metric(
            X=X[:, input_idx][self.initial_intervals[segment]],
            y=y[:, output_idx][self.initial_intervals[segment]],
            delay=self.delay,
            cc_alpha=self.cc_alpha,
        )

        self.miso_correlations["segment" + "_" + str(segment)][output_idx_name][
            input_idx_name
        ] = ncc
