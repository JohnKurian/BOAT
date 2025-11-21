import pandas as pd
from nbeats_forecast import NBeats
import numpy as np

from metrics import calculate_metrics




data = pd.read_csv('datasets/weather_energy_hourly.csv')
data = data.avg_energy.values        #univariate time series data of shape nx1 (numpy array)
data = data.reshape(-1,1)
print(data.shape)


def load_nbeats_data(dataset_location, predictor):
    data = pd.read_csv(dataset_location)

    data = data[predictor].values  # univariate time series data of shape nx1 (numpy array)
    data = (data - np.min(data)) / np.ptp(data)
    data = data.reshape(-1, 1)
    return data

def nbeats_model(dataset_location, predictor, forecasting_horizon, cfg, exp_id):
    data = load_nbeats_data(dataset_location, predictor)
    model = NBeats(data=data, period_to_forecast=forecasting_horizon)
    model.fit()
    forecast = model.predict()
    rmse, r2 = calculate_metrics(data[-forecasting_horizon:], forecast, forecasting_horizon)
    return  r2, rmse, data[-forecasting_horizon:].T, forecast.T
