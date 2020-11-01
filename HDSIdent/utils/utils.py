import pandas as pd
import numpy as np
from typing import Union
from typing import Tuple


def verify_data(
    X: Union[pd.DataFrame, np.array], y: Union[pd.DataFrame, np.array]
) -> Tuple[np.array, np.array, list, list]:
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
