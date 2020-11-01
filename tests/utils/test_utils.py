from pytest import fixture
import numpy as np
import pandas as pd
from HDSIdent.utils.utils import verify_data

@fixture
def dataframe_sample_1():
    """SISO Dataframe Without Column Names"""
    X = np.linspace(1,100,1000)
    y = np.sin(X)

    return (pd.DataFrame(X), pd.DataFrame(y))

@fixture
def dataframe_sample_2():
    """SISO Dataframe With Column Names"""
    X = np.linspace(1,100,1000)
    y = np.sin(X)

    X = pd.DataFrame(X)
    X.columns = ["X Test Signal"]

    y = pd.DataFrame(y)
    y.columns = ["y Test Signal"]

    return (X, y)

@fixture
def dataframe_sample_3():
    """MIMO Dataframe With Column Names"""
    X1 = np.linspace(1,100,1000).reshape(-1,1)
    X2 = (np.linspace(1,100,1000)*2 + 1).reshape(-1,1)

    y1 = np.sin(X1).reshape(-1,1)
    y2 = np.sin(X2).reshape(-1,1)

    X = pd.DataFrame(np.concatenate((X1,X2),axis=1))
    X.columns = ["X1 Test Signal", "X2 Test Signal"]

    y = pd.DataFrame(np.concatenate((y1,y2),axis=1))
    y.columns = ["y1 Test Signal", "y2 Test Signal"]

    return (X, y)

@fixture
def dataframe_sample_4():
    """MIMO Dataframe Without Column Names"""
    X1 = np.linspace(1,100,1000).reshape(-1,1)
    X2 = (np.linspace(1,100,1000)*2 + 1).reshape(-1,1)

    y1 = np.sin(X1).reshape(-1,1)
    y2 = np.sin(X2).reshape(-1,1)

    X = pd.DataFrame(np.concatenate((X1,X2),axis=1))

    y = pd.DataFrame(np.concatenate((y1,y2),axis=1))

    return (X, y)

@fixture
def array_sample_1():
    """SISO Array"""
    X = np.linspace(1,100,1000)
    y = np.sin(X)

    return (X, y)

@fixture
def array_sample_2():
    """MIMO Array"""
    X1 = np.linspace(1,100,1000).reshape(-1,1)
    X2 = (np.linspace(1,100,1000)*2 + 1).reshape(-1,1)

    y1 = np.sin(X1).reshape(-1,1)
    y2 = np.sin(X2).reshape(-1,1)

    X = np.concatenate((X1, X2), axis=1)
    y = np.concatenate((y1, y2), axis=1)

    return (X, y)

@fixture
def list_sample():
    """SISO List"""
    X = np.linspace(1,100,1000)
    y = np.sin(X)

    return (list(X), list(y))

def test_verify_data(dataframe_sample_1,
                     dataframe_sample_2,
                     dataframe_sample_3,
                     dataframe_sample_4,
                     array_sample_1,
                     array_sample_2,
                     list_sample):

    #Case 1: SISO Dataframe without columns
    X, y, X_cols, y_cols = verify_data(dataframe_sample_1[0],
                                       dataframe_sample_1[1])

    assert (X == dataframe_sample_1[0].values.reshape(-1,1)).all()
    assert (y == dataframe_sample_1[1].values.reshape(-1,1)).all()
    assert X_cols == 0
    assert y_cols == 0

    #Case 2: SISO Dataframe with columns
    X, y, X_cols, y_cols = verify_data(dataframe_sample_2[0],
                                       dataframe_sample_2[1])

    assert (X == dataframe_sample_2[0].values.reshape(-1,1)).all()
    assert (y == dataframe_sample_2[1].values.reshape(-1,1)).all()
    assert X_cols == ["X Test Signal"]
    assert y_cols == ["y Test Signal"]

    #Case 3: Multivariable Dataframe with columns
    X, y, X_cols, y_cols = verify_data(dataframe_sample_3[0],
                                       dataframe_sample_3[1])

    assert (X[:,0] == dataframe_sample_3[0].iloc[:,0]).all()
    assert (y[:, 0] == dataframe_sample_3[1].iloc[:,0]).all()
    assert (X[:,1] == dataframe_sample_3[0].iloc[:,1]).all()
    assert (y[:, 1] == dataframe_sample_3[1].iloc[:,1]).all()
    assert (X_cols == ["X1 Test Signal", "X2 Test Signal"]).all()
    assert (y_cols == ["y1 Test Signal", "y2 Test Signal"]).all()

    #Case 4: Multivariable Dataframe without columns
    X, y, X_cols, y_cols = verify_data(dataframe_sample_4[0],
                                       dataframe_sample_4[1])

    assert (X[:,0] == dataframe_sample_4[0].iloc[:,0]).all()
    assert (y[:, 0] == dataframe_sample_4[1].iloc[:,0]).all()
    assert (X[:,1] == dataframe_sample_4[0].iloc[:,1]).all()
    assert (y[:, 1] == dataframe_sample_4[1].iloc[:,1]).all()
    assert (X_cols == [0, 1]).all()
    assert (y_cols == [0, 1]).all()

    #Case 5: SISO Array
    X, y, X_cols, y_cols = verify_data(array_sample_1[0],
                                       array_sample_1[1])

    assert (X == array_sample_1[0].reshape(-1,1)).all()
    assert (y == array_sample_1[1].reshape(-1, 1)).all()
    assert X_cols == None
    assert y_cols == None

    #Case 6: SISO Array
    X, y, X_cols, y_cols = verify_data(array_sample_2[0],
                                       array_sample_2[1])

    assert (X[:,0] == array_sample_2[0][:,0]).all()
    assert (X[:,1] == array_sample_2[0][:,1]).all()
    assert (y[:,0] == array_sample_2[1][:,0]).all()
    assert (y[:,1] == array_sample_2[1][:,1]).all()
    assert X_cols == None
    assert y_cols == None

    #Case 7: SISO List
    try:
        X, y, X_cols, y_cols = verify_data(list_sample[0],
                                           list_sample[1])
    except Exception as e:
        assert e.args[0] == "Input data must be a pandas dataframe or a numpy array"