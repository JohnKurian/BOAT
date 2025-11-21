import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import ReducedRegressionForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from boat.utils.metrics import calculate_metrics

def load_data(dataset_location: str, predictor: str, forecasting_horizon: int) -> Tuple[pd.Series, pd.Series, ForecastingHorizon]:
    """
    Load data and split into train/test sets for forecasting.

    Args:
        dataset_location (str): Path to the dataset CSV file.
        predictor (str): Name of the column to predict.
        forecasting_horizon (int): Number of steps to forecast.

    Returns:
        Tuple[pd.Series, pd.Series, ForecastingHorizon]: Train series, test series, and forecasting horizon.
    """
    dataset = pd.read_csv(dataset_location)
    y = dataset[predictor]
    # Normalize
    y = (y - np.min(y)) / np.ptp(y)

    y_train, y_test = temporal_train_test_split(y, test_size=forecasting_horizon)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    return y_train, y_test, fh

def rf_regression(dataset_location: str, predictor: str, forecasting_horizon: int, cfg: Dict[str, Any], exp_id: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run Random Forest Regression forecast.

    Args:
        dataset_location (str): Path to dataset.
        predictor (str): Target column name.
        forecasting_horizon (int): Forecast horizon.
        cfg (Dict[str, Any]): Configuration dictionary containing 'window_length'.
        exp_id (str): Experiment ID (unused but kept for interface consistency).

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: R2, RMSE, y_test values, y_pred values.
    """
    y_train, y_test, fh = load_data(dataset_location, predictor, forecasting_horizon)
    np.random.seed(1)
    regressor = RandomForestRegressor()
    np.random.seed(1)
    forecaster = ReducedRegressionForecaster(regressor, window_length=cfg['window_length'])
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    rmse, r2 = calculate_metrics(y_test, y_pred, forecasting_horizon)
    return r2, rmse, y_test.values, y_pred.values

def kn_regression(dataset_location: str, predictor: str, forecasting_horizon: int, cfg: Dict[str, Any], exp_id: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run K-Nearest Neighbors Regression forecast.

    Args:
        dataset_location (str): Path to dataset.
        predictor (str): Target column name.
        forecasting_horizon (int): Forecast horizon.
        cfg (Dict[str, Any]): Configuration dictionary containing 'neighbours' and 'window_length'.
        exp_id (str): Experiment ID.

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: R2, RMSE, y_test values, y_pred values.
    """
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
    return r2, rmse, y_test.values, y_pred.values

def theta_forecaster(dataset_location: str, predictor: str, forecasting_horizon: int, cfg: Dict[str, Any], exp_id: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run Theta Forecaster.

    Args:
        dataset_location (str): Path to dataset.
        predictor (str): Target column name.
        forecasting_horizon (int): Forecast horizon.
        cfg (Dict[str, Any]): Configuration dictionary containing 'sp'.
        exp_id (str): Experiment ID.

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: R2, RMSE, y_test values, y_pred values.
    """
    y_train, y_test, fh = load_data(dataset_location, predictor, forecasting_horizon)
    np.random.seed(1)
    forecaster = ThetaForecaster(sp=cfg['sp'])  # seasonal periodicity
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    rmse, r2 = calculate_metrics(y_test, y_pred, forecasting_horizon)
    return r2, rmse, y_test.values, y_pred.values
