import pandas as pd
import numpy as np
import math
import copy
from collections import defaultdict
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns


class PettittMethod(object):
    """
    Performs signal segmentation using the pettit non-parametric
    method [Pettitt, A.N., 1979. A non-parametric approach to the
            change-point problem. Appl. Stat. 28, 126–135].

    Arguments:
        alpha: the statistical significance level;
        min_length_to_split: the minimum length an interval must have
        to produce a new change-point;
        split_size: the maximum length an interval can have. If an interval
        is larger than this value, it will be further divided.
        n_jobs: the number of CPUs to use;
        verbose: the degree of verbosity (going from 0 to 10).

    ----------------------------- REFERENCES -------------------------------
    This method is proposed in the following works:

       Pettitt, A.N., 1979. A non-parametric approach to the
       change-point problem. Appl. Stat. 28, 126–135.

       WANG, J. et al. Searching historical data segments for process
       identification in feedback control loops. Computers and Chemical
       Engineering, v. 112, n. 6, p. 6–16, 2018.
    """

    def __init__(
        self, alpha, min_length_to_split=0, split_size=200, n_jobs=-1, verbose=0
    ):

        self.df_cols = None

        if alpha is None:
            self.alpha = 0.05
        else:
            self.alpha = alpha

        self.min_length_to_split = min_length_to_split
        self.split_size = split_size
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Initialize Internal Variables
        self.initial_intervals = {}
        self.change_points = None
        self.segments = None

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
        if X is not None:
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
                raise Exception(
                    "Input data must be a pandas dataframe or a numpy array"
                )
        else:
            X_cols = None

        if y is not None:
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
                raise Exception(
                    "Input data must be a pandas dataframe or a numpy array"
                )
        else:
            y_cols = None

        return X, y, X_cols, y_cols

    def _initialize_segments(self, X, X_cols, y, y_cols):
        """
        Make initial segments for each signal.

        Aruments:
            X: the input data matrix. Each column corresponds to a signal.
            segments_dict: a dictionary having the signals indexes as a keys;
        """

        if X is not None:
            for input_idx in range(0, X.shape[1]):
                if X_cols is not None:
                    input_idx_name = X_cols[input_idx]
                else:
                    input_idx_name = "input" + "_" + str(input_idx)

                self.segments[input_idx_name] = [list(range(len(X[:, input_idx])))]

        if y is not None:
            for output_idx in range(0, y.shape[1]):
                if y_cols is not None:
                    output_idx_name = y_cols[output_idx]
                else:
                    output_idx_name = "output" + "_" + str(output_idx)

                self.segments[output_idx_name] = [list(range(len(y[:, output_idx])))]

    def _signal_stat(self, X, segment):
        """
        Creates the V statistic, from the following computation:
        Ut,T = Ut-1,T + Vt,T
        Vt,T = sum_{j=1}^{T}sgn(Xt - Xj), for t = 1,2,...,T

        Notice that T = len(segment)

        Arguments:
            X: the input data matrix. Each column corresponds to a signal.
            segment: the running segment.

        Output:
            V: the difference statistic
        """
        V = [
            np.sum(np.sign(X[segment[idx]] - X[segment])) for idx in range(len(segment))
        ]

        return V

    def _mann_whitney(self, V):
        """
        Computes the mann-whitney U statistic comming from the following computation:
        Ut,T = Ut-1,T + Vt,T
        Vt,T = sum_{j=1}^{T}sgn(Xt - Xj), for t = 1,2,...,T

        Notice that T = len(segment)

        Arguments:
            V: the difference statistic

        Output:
            U: the mann-whitney statistic
        """

        U = np.cumsum(V)

        return U

    def _cusum_mann_whitney(self, segment, U):
        """
        This function makes the optimization suggested by Pettitt (1979):

        tau = argmax(abs(U))

        Arguments:
            segment: the running segment
            U: the mann-whitney statistic

        Output:
            tau: the optimum segment index
        """
        tau_idx = np.argmax(np.abs(U))
        tau = segment[tau_idx]

        return tau

    def _calculate_p_value(self, U):
        """
        This function computes an estimative for the
        mann-whitney statistic p-value. Notice that if
        the p-value is lower than the given statistical
        significance level, then the changing-point can be
        consider significant.

        Arguments:
            U: the mann-whitney statistic

        Output:
            p_value: the p_value associated with the optimization
        """
        K = len(U)
        p_value = 2 * np.exp(
            (-6 * np.power(np.max(np.abs(U)), 2)) / (np.power(K, 2) + np.power(K, 3))
        )

        return p_value

    def _select_significant_tau(self, p_value_arr, tau_arr, alpha):
        """
        This function takes all the running changing-points and its
        corresponding estimated p-values and select those whose p-values
        are lower than the significance level provided.

        Arguments:
            p_value_arr: an array of p-values associated with each change-point
            tau_arr: an array of changing_points
            alpha: the significance value provided

        Output:
            sig_tau_arr: the significant changing-points indexes
        """
        p_args = np.where(np.array(p_value_arr) < alpha)[0]
        if p_args.size > 0:
            sig_tau_arr = list(np.sort(np.array(tau_arr)[p_args]))
            return sig_tau_arr
        else:
            return None

    def _update_change_points(self, X, segment):
        """
        This function computes the required statistics to perform
        the Pettitt non-parametric test and returns an array of
        changing-points with its corresponding p-values.

        Arguments:
            X: the input data matrix. Each column corresponds to a signal..
            segment: the running data segment.

        Output:
            A tuple of the form (p_value, tau), where tau is the changing-point.
        """
        if len(segment) > self.min_length_to_split:

            V = self._signal_stat(X, segment)
            U = self._mann_whitney(V)
            tau = self._cusum_mann_whitney(segment, U)
            p_value = self._calculate_p_value(U)

            if math.isnan(p_value) and len(segment) == len(X):
                p_value = 0

                if tau == 0 or tau == len(X):
                    tau = int(len(segment) / 2)

            elif math.isnan(p_value):
                p_value = np.inf

            if p_value == float("inf"):
                p_value = 0
                tau = int(tau + len(segment) / 2)

            return (p_value, tau)

        else:
            return (None, None)

    def _update_segments(self, sig_tau_arr, N):
        """
        This function updates the running segments. Given an array
        of significant changing-points, this functions splits the
        current segments into new segments according to each changing-point.

        Arguments:
            sig_tau_arr: an array of significant changing-poins
            N: the length of the entire running signal
        """
        new_segments_arr = []
        for idx in range(len(sig_tau_arr)):
            if len(sig_tau_arr) == 1:
                new_segments_arr.append(list(range(0, sig_tau_arr[idx])))
                new_segments_arr.append(list(range(sig_tau_arr[idx], N + 1)))
            elif len(sig_tau_arr) == 2:
                if idx == 0:
                    new_segments_arr.append(list(range(0, sig_tau_arr[idx])))
                else:
                    new_segments_arr.append(
                        list(range(sig_tau_arr[idx - 1], sig_tau_arr[idx]))
                    )
                    new_segments_arr.append(list(range(sig_tau_arr[idx], N + 1)))
            else:
                if idx == 0:
                    new_segments_arr.append(list(range(0, sig_tau_arr[idx])))
                elif idx == len(sig_tau_arr) - 1:
                    new_segments_arr.append(
                        list(range(sig_tau_arr[idx - 1], sig_tau_arr[idx]))
                    )
                    new_segments_arr.append(list(range(sig_tau_arr[idx], N + 1)))
                else:
                    new_segments_arr.append(
                        list(range(sig_tau_arr[idx - 1], sig_tau_arr[idx]))
                    )

        return new_segments_arr

    def _find_change_points(self, data, data_cols, data_type):

        for data_idx in range(0, data.shape[1]):

            # Take Column Name
            if data_cols is not None:
                data_idx_name = data_cols[data_idx]
            else:
                data_idx_name = data_type + "_" + str(data_idx)

            for _ in range(len(data[:, data_idx])):

                # Compute statistics and find significant changing-points
                executor = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

                change_point_taks = (
                    delayed(self._update_change_points)(data[:, data_idx], segment)
                    for segment in self.segments[data_idx_name]
                )

                change_point_taks_output = [
                    value
                    for value in list(executor(change_point_taks))
                    if (value[0] is not None and value[1] is not None)
                ]

                # Verify if no change-point was found
                if change_point_taks_output != []:
                    p_value_arr, tau_arr = list(
                        map(list, zip(*change_point_taks_output))
                    )
                else:
                    p_value_arr, tau_arr = None, None

                # If there is at least one significant change-point, update segments
                if (p_value_arr is not None) and (tau_arr is not None):
                    # Take significant change-point indexes
                    sig_tau_arr = self._select_significant_tau(
                        p_value_arr, tau_arr, self.alpha
                    )
                    if self.verbose > 0:
                        print(f"Signal: {data_idx_name}")
                        print(
                            "Change-points: {}".format(
                                self.change_points[data_idx_name]
                            )
                        )
                    if sig_tau_arr is not None:
                        self.change_points[data_idx_name] += sig_tau_arr
                        self.change_points[data_idx_name].sort()
                        self.segments[data_idx_name] = self._update_segments(
                            self.change_points[data_idx_name],
                            len(data[:, data_idx]) - 1,
                        )
                    else:
                        break
                else:
                    break

    def _split_long_intervals(self, data, data_cols, data_type):
        """
        This function splits too long intervals according
        to the provided value of split_size.
        """
        for col_idx in range(0, data.shape[1]):

            if data_cols is None:
                data_idx_name = data_type + "_" + str(col_idx)
            else:
                data_idx_name = data_cols[col_idx]

            divided_intervals = []
            for value in self.segments[data_idx_name]:
                if len(value) < self.split_size:
                    divided_intervals.append(value)
                else:
                    divided_intervals += list(
                        np.array_split(
                            np.array(value), math.ceil(len(value) / self.split_size)
                        )
                    )

            divided_intervals = [list(x) for x in divided_intervals]
            self.initial_intervals[data_idx_name] = copy.deepcopy(divided_intervals)

    def fit(self, X=None, y=None):
        """"""
        # Initialize Internal Variables
        self.change_points = defaultdict(list)
        self.segments = defaultdict(list)

        # Verify data format
        X, y, X_cols, y_cols = self._verify_data(X, y)

        # segments initialization
        self._initialize_segments(X=X, X_cols=X_cols, y=y, y_cols=y_cols)

        # Find Change-points
        ##Input Data
        if X is not None:
            self._find_change_points(data=X, data_cols=X_cols, data_type="input")
            self._split_long_intervals(data=X, data_cols=X_cols, data_type="input")

        ##Output Data
        if y is not None:
            self._find_change_points(data=y, data_cols=y_cols, data_type="output")
            self._split_long_intervals(data=y, data_cols=y_cols, data_type="output")

        return self.initial_intervals

    def _plot_data(self, data, data_cols, data_type, divided_data, show_intervals):

        for data_idx in range(0, data.shape[1]):

            # Take Column Name
            if data_cols is not None:
                data_idx_name = data_cols[data_idx]
            else:
                data_idx_name = data_type + "_" + str(data_idx)

            if divided_data:
                intervals_arr = self.initial_intervals[data_idx_name]
            else:
                intervals_arr = self.segments[data_idx_name]

            sns.set_style("darkgrid")
            plt.figure(figsize=(15, 5))
            plt.plot(data[:, data_idx], zorder=1, linewidth=0.8, color="coral")
            plt.title(
                f"Pettitt Non-Parametric Change Points and Intervals for {data_idx_name}",
                fontsize=20,
                fontweight="bold",
            )
            plt.ylabel("Signal Amplitude", fontsize=20)
            plt.xlabel("Discrete Samples", fontsize=20)
            plt.xticks(fontsize=20, color="black")
            plt.yticks(fontsize=20, color="black")

            for idx in range(0, len(intervals_arr) - 1):
                plt.scatter(
                    np.max(intervals_arr[idx]),
                    data[np.max(intervals_arr[idx]), data_idx],
                    marker="x",
                    s=50,
                    color="black",
                    zorder=3,
                )

            plt.show()

    def plot_change_points(
        self, X=None, y=None, divided_data=False, show_intervals=False
    ):
        """
        Plots all found change points and its corresponding
        intervals.

        Arguments:
            verbose: verbose level as in joblib library
            n_jobs: the number of threads as in joblib library
        """
        # Verify data format
        X, y, X_cols, y_cols = self._verify_data(X, y)

        # If data if not done yet
        if self.change_points is None:
            self.fit(X=X, y=y)

        # Plot input data
        if X is not None:
            self._plot_data(
                data=X,
                data_cols=X_cols,
                data_type="input",
                divided_data=divided_data,
                show_intervals=show_intervals,
            )

        # Plot Output Data
        if y is not None:
            self._plot_data(
                data=y,
                data_cols=y_cols,
                data_type="output",
                divided_data=divided_data,
                show_intervals=show_intervals,
            )


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
