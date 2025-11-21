import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pandas as pd


from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    EnsembleForecaster,
    ReducedRegressionForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)

from sktime.forecasting.theta import ThetaForecaster
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from metrics import calculate_metrics




def load_data(dataset_location, predictor, forecasting_horizon):
    dataset = pd.read_csv(dataset_location)
    y = dataset[predictor]
    y = (y - np.min(y)) / np.ptp(y)

    y_train, y_test = temporal_train_test_split(y, test_size=forecasting_horizon)

    fh = ForecastingHorizon(y_test.index, is_relative=False)
    return y_train, y_test, fh




def rf_regression(dataset_location, predictor, forecasting_horizon, cfg, exp_id):
    y_train, y_test, fh = load_data(dataset_location, predictor, forecasting_horizon)
    np.random.seed(1)
    regressor = RandomForestRegressor()
    np.random.seed(1)
    forecaster = ReducedRegressionForecaster(regressor, window_length=cfg['window_length'])
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    rmse, r2 = calculate_metrics(y_test, y_pred, forecasting_horizon)
    return r2, rmse, y_test, y_pred



def kn_regression(dataset_location, predictor, forecasting_horizon, cfg, exp_id):
    y_train, y_test, fh = load_data(dataset_location, predictor, forecasting_horizon)
    np.random.seed(1)
    regressor = KNeighborsRegressor(n_neighbors=cfg["neighbours"])
    np.random.seed(1)
    forecaster = ReducedRegressionForecaster(
        regressor=regressor, window_length=cfg["window_length"], strategy="recursive"
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    rmse, r2 = calculate_metrics(y_test, y_pred, forecasting_horizon)
    return r2, rmse, y_test, y_pred


def theta_forecaster(dataset_location, predictor, forecasting_horizon, cfg, exp_id):
    y_train, y_test, fh = load_data(dataset_location, predictor, forecasting_horizon)
    np.random.seed(1)
    forecaster = ThetaForecaster(sp=cfg['sp'])  # monthly seasonal periodicity
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    rmse, r2 = calculate_metrics(y_test, y_pred, forecasting_horizon)
    return r2, rmse, y_test, y_pred