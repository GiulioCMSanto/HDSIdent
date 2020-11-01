from HDSIdent.data_treatment.data_preprocessing import Preprocessing
from pytest import fixture
import numpy as np
import pandas as pd

@fixture
def data_sample():
    X = np.linspace(1,100,1000)
    y = np.sin(X)*2 + 1

    return (X, y)

@fixture
def dataframe_sample():
    X = np.linspace(1,100,1000)
    X = pd.DataFrame(X, columns=['X Test Signal'])

    y = np.sin(X)*2 + 1
    y = pd.DataFrame(y, columns=['Y Test Signal'])

    return (X, y)

    return (X, y)

class TestPreprocessing:

    def test_case_1(self, data_sample):
        """MinMaxScaler"""

        #[0,1] Scaling
        pp_1 = Preprocessing(
                scaler='MinMaxScaler',
                feature_range=(0,1),
                k=10)

        X_clean, Y_clean = pp_1.fit_transform(
                                    X=data_sample[0],
                                    y=data_sample[1])

        assert np.max(X_clean) == 1
        assert np.min(X_clean) == 0
        assert np.max(Y_clean) == 1
        assert np.min(Y_clean) == 0
        assert np.mean(X_clean) == 0.5
        assert X_clean.shape[0] == data_sample[0].shape[0] - pp_1.k

        #[-0.5, 0.5] Scaling
        pp_1 = Preprocessing(
                scaler='MinMaxScaler',
                feature_range=(-0.5,0.5),
                k=20)

        X_clean, Y_clean = pp_1.fit_transform(
                                    X=data_sample[0],
                                    y=data_sample[1])

        assert np.around(np.max(X_clean),3) == 0.5
        assert np.around(np.min(X_clean),3) == -0.5
        assert np.around(np.max(Y_clean),3) == 0.5
        assert np.around(np.min(Y_clean),3) == -0.5
        assert np.around(np.mean(X_clean),3) == 0.0
        assert X_clean.shape[0] == data_sample[0].shape[0] - pp_1.k

    def test_case_2(self, data_sample):
        """StandardScale"""
        pp_2 = Preprocessing()

        X_clean, Y_clean = pp_2.fit_transform(
                                    X=data_sample[0],
                                    y=data_sample[1])

        mean_X = np.mean(data_sample[0][pp_2.k:])
        std_X = np.std(data_sample[0][pp_2.k:])
        z_X = (data_sample[0][pp_2.k:]-mean_X)/std_X

        mean_y = np.mean(data_sample[1][pp_2.k:])
        std_y = np.std(data_sample[1][pp_2.k:])
        z_y = (data_sample[1][pp_2.k:]-mean_y)/std_y

        assert (np.squeeze(np.around(X_clean,5)) == np.round(z_X,5)).all()
        assert (np.squeeze(np.around(Y_clean, 5)) == np.round(z_y, 5)).all()
        assert X_clean.shape[0] == data_sample[0].shape[0] - pp_2.k

    def test_case_3(self, data_sample):
        """Butterworth Filter"""
        pp_3 = Preprocessing(W=0.5)

        X_clean, Y_clean = pp_3.fit_transform(
                                    X=data_sample[0],
                                    y=data_sample[1])

        assert np.around(np.max(X_clean),2) == 1.73
        assert np.around(np.min(X_clean), 2) == -1.73
        assert X_clean.shape[0] == data_sample[0].shape[0] - pp_3.k

    def test_case_4(self, dataframe_sample):
        """DataFrame Test"""
        pp_4 = Preprocessing()

        X_clean, Y_clean = pp_4.fit_transform(
                                    X=dataframe_sample[0],
                                    y=dataframe_sample[1])

        assert (X_clean.columns == ['X Test Signal']).all()
        assert (Y_clean.columns == ['Y Test Signal']).all()

