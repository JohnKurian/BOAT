import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from nbeats_forecast import NBeats

from boat.utils.metrics import calculate_metrics

def load_nbeats_data(dataset_location: str, predictor: str) -> np.ndarray:
    """
    Load and normalize data for NBeats.
    """
    data = pd.read_csv(dataset_location)
    data = data[predictor].values
    data = (data - np.min(data)) / np.ptp(data)
    data = data.reshape(-1, 1)
    return data

def nbeats_model(dataset_location: str, predictor: str, forecasting_horizon: int, cfg: Dict[str, Any], exp_id: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Train and evaluate an NBeats model.
    """
    data = load_nbeats_data(dataset_location, predictor)
    model = NBeats(data=data, period_to_forecast=forecasting_horizon)
    model.fit()
    forecast = model.predict()
    
    # NBeats predict returns forecast for the horizon
    # We compare against the last 'forecasting_horizon' points of the data
    # Note: data is normalized here.
    
    rmse, r2 = calculate_metrics(data[-forecasting_horizon:], forecast, forecasting_horizon)
    
    return r2, rmse, data[-forecasting_horizon:].T, forecast.T
