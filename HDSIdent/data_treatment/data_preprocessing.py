from HDSIdent.utils.utils import verify_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from collections import defaultdict
from scipy import signal
import pandas as pd
import numpy as np


class Preprocessing(object):
    """
    Performs preprocessing transformations in a given dataframe.
    The following transformations are applied:

        - An array of bad data (data with missing values) is created;
        - Data Scaling (Either StandardScaler or MinMaxScaler);
        - Removal of first samples to avoid deflection;
        - Butterworth low-pass filtering.

    Arguments:
        scaler: the sklearn scaler type to be used (MinMaxScaler or StandardScaler);
        feature_range: the normalization feature range as a tuple (min, max);
        k: the number of initial samples to be removed;
        W: the input cut-off frequencies as a list [wmin, wmax];
        N: Butterworth filter order;
        Ts: the sampling frequency of the digital filter (Default = 1.0 seconds)/

    --------------------------------- REFERENCES --------------------------------------

    The preprocessing methods adopted were based on the following references:

        PERETZKI, D. et al. Data mining of historic data for process identification.
        In: Proceedings of the 2011 AIChE Annual Meeting, p. 1027–1033, 2011.

        BITTENCOURT, A. C. et al. An algorithm for finding process identification
        intervals from normal operating data. Processes, v. 3, p. 357–383, 2015.

        PATEL, A. Data Mining of Process Data in Mutlivariable Systems.
        Degree project in electrical engineering — Royal Institute of Technology,
        Stockholm, Sweden, 2016.

        FACELI, K. et al. Inteligência Artificial: Uma Abordagem de Aprendizado de
        Máquina. Rio de Janeiro, Brasil: LTC, 2017. (In portuguese)
    """

    def __init__(
        self, scaler="StandardScaler", feature_range=(0, 1), k=10, W=None, N=1, Ts=1
    ):

        self.scaler = scaler
        self.feature_range = feature_range
        self.k = k
        self.W = W
        self.N = N
        self.Ts = Ts

    def _scale(self, data):
        """
        Scales the dataframe according to the scaler type
        provided.
        """
        if self.scaler == "MinMaxScaler":
            scl = MinMaxScaler(feature_range=self.feature_range)
            data = scl.fit_transform(data)
        elif self.scaler == "StandardScaler":
            scl = StandardScaler()
            data = scl.fit_transform(data)
        else:
            raise Exception("Only MinMaxScaler and StandardScaler are accepted")

        return data

    def _remove_first_samples(self, data):
        """
        Removes the initial samples to avoid deflection.
        """
        return data[self.k :, :]

    def _lowpass_filter(self, data):
        """
        Performs a butterworth lowpass filtering
        to remove high frequency noise.
        """
        butt_mtrx = np.empty(shape=data.shape)

        num, den = signal.butter(N=self.N, Wn=self.W, btype="low", analog=False)
        e = signal.TransferFunction(num, den, dt=self.Ts)

        for col in range(data.shape[1]):
            t_in = np.arange(0, data.shape[0], 1)
            t_out, butt_arr = signal.dlsim(e, data[:, col], t=t_in)
            butt_mtrx[:, col] = butt_arr.reshape(-1, 1)[:, 0]

        return butt_mtrx

    def _defined_bad_data(self, X, X_cols, y, y_cols):
        """
        For each signal, defines an array of indexes
        whith the value of 0 for the indexes where
        the data is not null and with the value of
        1 for the indexes where the data is null.

        A dictionary is defined such data each key
        corresponds to a particular signal and each
        value corresponds to an array of indexes of
        bad data for the corresponding signal.

        Arguments:
            X: a matrix of input signals. Each signal is a column;
            X_cols: the input data columns in case they are provided;
            y: a matrix of output signals. Each signal is a column;
            y_cols: the output data columns in case they are provided.
        """
        self.bad_data_dict = defaultdict(list)

        # Define input bad data
        for input_idx in range(0, X.shape[1]):
            if X_cols is not None:
                input_idx_name = X_cols[input_idx]
            else:
                input_idx_name = "input" + "_" + str(input_idx)

            self.bad_data_dict[input_idx_name] = np.zeros(len(X[:, input_idx]))
            self.bad_data_dict[input_idx_name][
                np.argwhere(np.isnan(X[:, input_idx]))
            ] = 1

        # Define output bad data
        for output_idx in range(0, y.shape[1]):
            if y_cols is not None:
                output_idx_name = y_cols[output_idx]
            else:
                output_idx_name = "output" + "_" + str(output_idx)

            self.bad_data_dict[output_idx_name] = np.zeros(len(y[:, output_idx]))
            self.bad_data_dict[output_idx_name][
                np.argwhere(np.isnan(y[:, output_idx]))
            ] = 1

    def fit_transform(self, X, y):
        """
        Performs all steps in the preprocessing class for the
        given data, which includes:

        1) Defining the sequences of bad data (data with null values);
        2) Removing first k samples;
        3) Scaling the data;
        4) Applying a lowpass filter.

        Arguments:
            X: the input data in pandas dataframe format or numpy array
            y: the output data in pandas dataframe format or numpy array
        """
        X, y, X_cols, y_cols = verify_data(X, y)

        # Define Bad Data
        self._defined_bad_data(X=X, X_cols=X_cols, y=y, y_cols=y_cols)

        # Remove First Samples from Data
        X_aux = self._remove_first_samples(data=X)
        y_aux = self._remove_first_samples(data=y)

        # Scale Data
        data = np.concatenate([X_aux, y_aux], axis=1)
        data = self._scale(data=data)

        X_aux = data[:, : X.shape[1]]
        y_aux = data[:, X.shape[1] :]

        # Apply Lowpass Filter
        if self.W:
            X_aux = self._lowpass_filter(data=X_aux)
            y_aux = self._lowpass_filter(data=y_aux)

        # Create Dataframe
        if X_cols is not None:
            X_aux = pd.DataFrame(X_aux)
            X_aux.columns = X_cols

        if y_cols is not None:
            y_aux = pd.DataFrame(y_aux)
            y_aux.columns = y_cols

        return X_aux, y_aux


# See below the used libraries Licenses
# -------------------------------------

# Scikit-learn license
# --------------------

# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

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

# Scipy license
# -------------

# Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

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
