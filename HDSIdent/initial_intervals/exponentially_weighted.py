import math
import pandas as pd
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


class ExponentiallyWeighted(object):
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

    Reference works:
        PERETZKI, D. et al. Data mining of historic data for process identification.
        In: Proceedings of the 2011 AIChE Annual Meeting, p. 1027–1033, 2011.

        BITTENCOURT, A. C. et al. An algorithm for finding process identification
        intervals from normal operating data. Processes, v. 3, p. 357–383, 2015.

        WANG, J. et al. Searching historical data segments for process
        identification in feedback control loops. Computers and Chemical
        Engineering, v. 112, n. 6, p. 6–16, 2018.
    """

    def __init__(
        self,
        forgetting_fact_v,
        forgetting_fact_u,
        sigma=None,
        H_u=None,
        H_v=None,
        min_input_coupling=1,
        min_output_coupling=1,
        num_previous_indexes=0,
        min_interval_length=None,
        split_size=None,
        n_jobs=-1,
        verbose=0,
    ):

        self.forgetting_fact_v = forgetting_fact_v
        self.forgetting_fact_u = forgetting_fact_u
        self.sigma = sigma
        self.H_u = H_u
        self.H_v = H_v
        self.min_input_coupling = min_input_coupling
        self.min_output_coupling = min_output_coupling
        self.num_previous_indexes = num_previous_indexes
        self.min_interval_length = min_interval_length
        self.split_size = split_size
        self.n_jobs = n_jobs
        self.verbose = verbose

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

    def _initialize_internal_variables(self, X):
        """
        This function initializes the interval variables.
        """
        self.unified_intervals = defaultdict(list)
        self.intervals = defaultdict(list)
        self._mu_k_arr = None
        self._v_k_arr = None
        self._mu_k = np.array([])
        self._v_k = np.array([])
        self._is_interval = [False] * X.shape[1]
        self._init_idx = [0] * X.shape[1]
        self._final_idx = [0] * X.shape[1]
        self._criteria = None

        self._mu_k_1 = np.mean(X[:100, :], axis=0)
        self._v_k_1 = np.var(X[:100, :], axis=0)

        if self.sigma is None:
            self.sigma = np.std(X, axis=0)

        if self.H_u is None:
            self.H_u = 5 * self.sigma

        if self.H_v is None:
            self.H_v = 5 * self.sigma

        if type(self.H_u) == list:
            self.H_u = np.array(self.H_u)
        if type(self.H_v) == list:
            self.H_u = np.array(self.H_u)

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

        self._mu_k = (
            self.forgetting_fact_u * X[idx, :]
            + (1 - self.forgetting_fact_u) * self._mu_k_1
        )
        self._v_k = ((2 - self.forgetting_fact_u) / 2) * (
            self.forgetting_fact_v * (X[idx, :] - self._mu_k) ** 2
            + (1 - self.forgetting_fact_v) * self._v_k_1
        )

        self._mu_k_1 = self._mu_k
        self._v_k_1 = self._v_k

        return (self._mu_k, self._v_k)

    def _search_for_change_points(self, X, idx, col, criteria):
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

        # Change-point conditions
        if criteria == "average":
            condition = abs(self._mu_k_arr[idx, col]) >= self.H_u[col]
        elif criteria == "variance":
            condition = abs(self._v_k_arr[idx, col]) >= self.H_v[col]
        else:
            condition = (abs(self._mu_k_arr[idx, col]) >= self.H_u[col]) and (
                abs(self._v_k_arr[idx, col]) >= self.H_v[col]
            )

        if condition:
            if not self._is_interval[col]:
                self._init_idx[col] = idx
                self._is_interval[col] = True
            elif idx == len(X) - 1 and self._is_interval[col]:
                self._is_interval[col] = False
                self._final_idx[col] = idx
                self.intervals[col].append([self._init_idx[col], self._final_idx[col]])
        elif self._is_interval[col]:
            self._is_interval[col] = False
            self._final_idx[col] = idx
            self.intervals[col].append([self._init_idx[col], self._final_idx[col]])

    def recursive_exponential_moving_average_and_variance(self, X):
        """
        Performs a recursive moving average/variance algorithm from past samples
        using a multithread approach.

        Output:
            self._mu_k_arr: the average filtered data for the given index
            self._v_k_arr: the variance filtered data for the given index
        """
        results = list(
            Parallel(n_jobs=self.n_jobs, require="sharedmem", verbose=self.verbose)(
                delayed(self._exponential_moving_average_and_variance)(X, idx)
                for idx in range(len(X))
            )
        )

        self._mu_k_arr, self._v_k_arr = list(zip(*results))
        self._mu_k_arr = np.stack(self._mu_k_arr, axis=0)
        self._v_k_arr = np.stack(self._v_k_arr, axis=0)

        return self._mu_k_arr, self._v_k_arr

    def change_points(self, X, criteria="variance"):
        """
        Searchs for change points in the filtered data and its
        corresponding intervals using a multithread approach.

        Arguments:
            criteria: the filter to be considered when looking for
                      a change-point (average, variance or both)
        """
        # Reset Intervals
        self.intervals = defaultdict(list)

        # Update Criteria
        self._criteria = criteria

        if (self._mu_k_arr is None) or (self._v_k_arr is None):
            self.recursive_exponential_moving_average_and_variance()

        Parallel(n_jobs=self.n_jobs, require="sharedmem", verbose=self.verbose)(
            delayed(self._search_for_change_points)(X, idx, col, criteria)
            for idx in range(len(X))
            for col in range(X.shape[1])
        )

        self._is_interval = [False] * X.shape[1]
        self._init_idx = [0] * X.shape[1]
        self._final_idx = [0] * X.shape[1]

        return self.intervals

    def _extend_previous_indexes(self):
        """
        This function allows an extension of each interval
        with previous index values. The number of indexes
        extended are provided in num_previous_indexes.
        """
        for key, interval_arr in self.intervals.items():
            for idx, interval in enumerate(interval_arr):

                min_val = np.min(interval)

                if (idx == 0) and (np.min(interval) - self.num_previous_indexes < 0):
                    min_val = 0
                elif (idx > 0) and (
                    (np.min(interval) - self.num_previous_indexes)
                    <= np.max(interval_arr[idx - 1])
                ):
                    min_val = np.max(interval_arr[idx - 1]) + 1
                else:
                    min_val = np.min(interval) - self.num_previous_indexes

                self.intervals[key][idx] = [min_val, np.max(interval)]

    def _create_indicating_sequence(self, X):
        """
        This function creates an indicating sequence, i.e., an array containing 1's
        in the intervals of interest and 0's otherwise, based on each interval obtained
        by the exponential weighted filter approach.

        Output:
            indicating_sequence: the indicating sequence
        """
        indicating_sequence = np.zeros(X.shape[0])
        for _, interval_arr in self.intervals.items():
            for interval in interval_arr:
                indicating_sequence[interval[0] : interval[1] + 1] = 1

        return indicating_sequence

    def _create_sequential_indicating_sequences(self, indicating_sequence):
        """
        This function gets the indicating sequence for a given data
        and creates the corresponding segments where the sequence
        contains consecutive values of 1. For example, the sequence
        [0,0,1,1,1,1,0,0,0,1,1,0,0,0] would result in two sequential
        sequences:

        1) Sequence formed by indexes [2,3,4,5]
        2) Sequence forme by indexes [9,10]

        Arguments:
            indicating_sequence: the data indicating sequence.

        Output:
            sequential_indicating_sequences: the sequential indicating sequence.
        """

        is_interval = False
        sequential_indicating_sequences = []
        aux_arr = []

        for idx in range(len(indicating_sequence)):

            if not is_interval and indicating_sequence[idx] == 1:
                is_interval = True

            if is_interval and indicating_sequence[idx] == 1:
                aux_arr.append(idx)

            if idx < len(indicating_sequence) - 1:
                if (
                    is_interval
                    and indicating_sequence[idx] == 1
                    and indicating_sequence[idx + 1] == 0
                ):

                    is_interval = False
                    sequential_indicating_sequences.append(aux_arr)
                    aux_arr = []
            else:
                if aux_arr != []:
                    sequential_indicating_sequences.append(aux_arr)

        return sequential_indicating_sequences

    def _label_intervals_with_input_output(self, X, X_cols, y, y_cols):
        """
        This function labels the intervals dictionary keys to discriminate
        the input and output variables. This is crucial to garantee the
        min_input_coupling and min_output_coupling conditions.

        Arguments:
            X: the input matrix. Each column corresponds to an input signal
            X_cols: the input signals column names
            y: the output matrix: Each column corresponds to an ouput signal
            y_cols: the output signals column names
        """

        labeled_intervals = defaultdict(dict)

        for input_idx in range(0, X.shape[1]):

            if X_cols is not None:
                input_idx_name = X_cols[input_idx]
            else:
                input_idx_name = "input" + "_" + str(input_idx)

            labeled_intervals["input"][input_idx_name] = self.intervals[input_idx]

        for output_idx in range(0, y.shape[1]):

            if y_cols is not None:
                output_idx_name = y_cols[output_idx]
            else:
                output_idx_name = "output" + "_" + str(output_idx)

            labeled_intervals["output"][output_idx_name] = self.intervals[
                X.shape[1] + output_idx
            ]

        return labeled_intervals

    def _get_final_intervals(self, labeled_intervals, global_sequence):
        """
        This function takes the global indicating sequences, i.e., the unified
        indicating sequence for all input and output signals and verfies if
        there is at least one input and one output valid indicating sequence inside
        each global indicating sequence.

        Arguments:
            global_sequence: the unified intervals for all input and output signals.
            labeled_intervals: the individual intervals for each input and output.
        """

        final_segment_indexes = []

        for segment_idx_arr in global_sequence:

            # Check if at least one input indicating sequence is in the correspondig global sequence
            input_count = 0
            for input_name in labeled_intervals["input"].keys():
                input_aux_count = 0
                for input_sequence in labeled_intervals["input"][input_name]:
                    if all(elem in segment_idx_arr for elem in input_sequence):
                        input_aux_count += 1
                if input_aux_count > 0:
                    input_count += 1

            # Check if at least one output indicating sequence is in the correspondig global sequence
            output_count = 0
            for output_name in labeled_intervals["output"].keys():
                output_aux_count = 0
                for output_sequence in labeled_intervals["output"][output_name]:
                    if all(elem in segment_idx_arr for elem in output_sequence):
                        output_aux_count += 1
                if output_aux_count > 0:
                    output_count += 1

            if (
                input_count >= self.min_input_coupling
                and output_count >= self.min_output_coupling
            ):

                final_segment_indexes.append(segment_idx_arr)

        return final_segment_indexes

    def _length_check(self):
        """
        This function checks the interval length
        according to the provided min_interval_length.
        Only intervals with length >= min_interval_length
        are returned.
        """
        final_intervals = {}

        for key, value in self.unified_intervals.items():
            if len(value) >= self.min_interval_length:
                final_intervals[key] = value

        return final_intervals

    def _split_data(self):
        """"""
        final_intervals = {}
        divided_intervals = []

        for key, value in self.unified_intervals.items():
            if len(value) < self.split_size:
                divided_intervals.append(value)
            else:
                divided_intervals += list(
                    np.array_split(
                        np.array(value), math.ceil(len(value) / self.split_size)
                    )
                )

        for key, interval in enumerate(divided_intervals):
            final_intervals[key] = list(interval)

        return final_intervals

    def fit(self, X, y):
        """
        This function performs the following routines:
            - Applies the recursive exponential moving average/variance
            - Compute the initial intervals (change-points)
            - Creates an indicating sequence, unifying input and output intervals
            - From the indicating sequence, creates a final unified interval

        Output:
            unified_intervals: the final unified intervals for the input and output signals
        """

        # Verify data format
        X, y, X_cols, y_cols = self._verify_data(X, y)

        # Create Matrix
        data = np.concatenate([X, y], axis=1)

        # Initialize Internal Variables
        self._initialize_internal_variables(X=data)

        # Apply Recursive Exponential Moving Average/Variance
        self.recursive_exponential_moving_average_and_variance(X=data)

        # Find change-points
        self.change_points(X=data)

        # Extend Intervals
        if self.num_previous_indexes > 0:
            self._extend_previous_indexes()

        # Make labeled intervals
        self.labeled_intervals = self._label_intervals_with_input_output(
            X=X, X_cols=X_cols, y=y, y_cols=y_cols
        )

        # Create Indicating Sequence
        indicating_sequence = self._create_indicating_sequence(X=data)

        # Create Global Sequence
        global_sequence = self._create_sequential_indicating_sequences(
            indicating_sequence=indicating_sequence
        )

        # Find intervals that respect min_input_coupling and min_output_coupling
        final_segment_indexes = self._get_final_intervals(
            labeled_intervals=self.labeled_intervals, global_sequence=global_sequence
        )

        self.unified_intervals = dict(
            zip(range(0, len(final_segment_indexes)), final_segment_indexes)
        )

        # Length Check
        if (self.min_interval_length is not None) and (self.min_interval_length > 1):
            self.unified_intervals = self._length_check()

        # Split Long Data
        if self.split_size:
            self.unified_intervals = self._split_data()

        return self.unified_intervals

    def plot_change_points(self, X, y):
        """
        Plots all found change points and its corresponding
        intervals.
        """
        # Verify data format
        X, y, X_cols, y_cols = self._verify_data(X, y)

        # Create Matrix
        data = np.concatenate([X, y], axis=1)
        df_cols = None

        if X_cols is not None and y_cols is not None:
            df_cols = list(X_cols) + list(y_cols)

        # Check if fit is needed
        try:
            self.intervals
        except:
            self.fit(X=X, y=y)

        for col in list(self.intervals.keys()):
            intervals_arr = self.intervals[col]

            sns.set_style("darkgrid")
            plt.figure(figsize=(15, 4))
            if self._criteria == "variance":
                plt.plot(self._v_k_arr[:, col], zorder=1, color="coral")
            elif self._criteria == "average":
                plt.plot(self._mu_k_arr[:, col], zorder=1, color="coral")
            else:
                plt.plot(
                    self._v_k_arr[:, col],
                    label="Variance Plot",
                    zorder=1,
                    color="coral",
                )
                plt.plot(
                    self._mu_k_arr[:, col],
                    label="Average Plot",
                    zorder=1,
                    color="coral",
                )
                plt.legend(fontsize=14)

            if df_cols is None:
                col_name = f"Signal {col}"
            else:
                col_name = f"Signal {df_cols[col]}"

            plt.title(
                f"Moving Average Change Points and Intervals for {col_name}",
                fontsize=18,
                fontweight="bold",
            )
            plt.ylabel("Signal Amplitude", fontsize=18)
            plt.xlabel("Discrete Samples", fontsize=18)
            plt.xticks(fontsize=18, color="black")
            plt.yticks(fontsize=18, color="black")

            color_rule = True
            color_arr = ["darkred", "darkmagenta"]
            for interval in intervals_arr:
                color_rule = not color_rule
                for idx in interval:
                    if self._criteria == "variance":
                        plt.scatter(
                            idx,
                            self._v_k_arr[:, col][idx],
                            marker="x",
                            s=50,
                            color=color_arr[color_rule],
                            zorder=2,
                        )
                        plt.axvline(x=idx, linestyle="--", color=color_arr[color_rule])
                    else:
                        plt.scatter(
                            idx,
                            self._mu_k_arr[:, col][idx],
                            marker="x",
                            s=50,
                            color=color_arr[color_rule],
                            zorder=2,
                        )
                        plt.axvline(x=idx, linestyle="--", color=color_arr[color_rule])
            plt.show()


# See below the used libraries Licenses
# -------------------------------------

# Joblib license
# --------------

# Copyright (c) 2008-2016, The joblib developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# Pandas license
# --------------

# Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
# All rights reserved.
#
# Copyright (c) 2011-2020, Open source contributors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# Numpy license
# -------------

# Copyright (c) 2005-2020, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# * Neither the name of the NumPy Developers nor the names of any
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# Matplotlib licence
# ------------------

# License agreement for matplotlib versions 1.3.0 and later
# =========================================================
#
# 1. This LICENSE AGREEMENT is between the Matplotlib Development Team
# ("MDT"), and the Individual or Organization ("Licensee") accessing and
# otherwise using matplotlib software in source or binary form and its
# associated documentation.
#
# 2. Subject to the terms and conditions of this License Agreement, MDT
# hereby grants Licensee a nonexclusive, royalty-free, world-wide license
# to reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use matplotlib
# alone or in any derivative version, provided, however, that MDT's
# License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
# 2012- Matplotlib Development Team; All Rights Reserved" are retained in
# matplotlib  alone or in any derivative version prepared by
# Licensee.
#
# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates matplotlib or any part thereof, and wants to
# make the derivative work available to others as provided herein, then
# Licensee hereby agrees to include in any such work a brief summary of
# the changes made to matplotlib .
#
# 4. MDT is making matplotlib available to Licensee on an "AS
# IS" basis.  MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
# IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
# DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
# FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
# WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.
#
# 5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
#  FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
# LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
# MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
# THE POSSIBILITY THEREOF.
#
# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.
#
# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between MDT and
# Licensee.  This License Agreement does not grant permission to use MDT
# trademarks or trade name in a trademark sense to endorse or promote
# products or services of Licensee, or any third party.
#
# 8. By copying, installing or otherwise using matplotlib ,
# Licensee agrees to be bound by the terms and conditions of this License
# Agreement.
#
# License agreement for matplotlib versions prior to 1.3.0
# ========================================================
#
# 1. This LICENSE AGREEMENT is between John D. Hunter ("JDH"), and the
# Individual or Organization ("Licensee") accessing and otherwise using
# matplotlib software in source or binary form and its associated
# documentation.
#
# 2. Subject to the terms and conditions of this License Agreement, JDH
# hereby grants Licensee a nonexclusive, royalty-free, world-wide license
# to reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use matplotlib
# alone or in any derivative version, provided, however, that JDH's
# License Agreement and JDH's notice of copyright, i.e., "Copyright (c)
# 2002-2011 John D. Hunter; All Rights Reserved" are retained in
# matplotlib  alone or in any derivative version prepared by
# Licensee.
#
# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates matplotlib  or any part thereof, and wants to
# make the derivative work available to others as provided herein, then
# Licensee hereby agrees to include in any such work a brief summary of
# the changes made to matplotlib.
#
# 4. JDH is making matplotlib  available to Licensee on an "AS
# IS" basis.  JDH MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
# IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, JDH MAKES NO AND
# DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
# FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
# WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.
#
# 5. JDH SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
#  FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
# LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
# MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
# THE POSSIBILITY THEREOF.
#
# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.
#
# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between JDH and
# Licensee.  This License Agreement does not grant permission to use JDH
# trademarks or trade name in a trademark sense to endorse or promote
# products or services of Licensee, or any third party.
#
# 8. By copying, installing or otherwise using matplotlib,
# Licensee agrees to be bound by the terms and conditions of this License
# Agreement.

# Seaborn license
# ---------------

# Copyright (c) 2012-2020, Michael L. Waskom
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the project nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
