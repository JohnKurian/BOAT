import pandas as pd
from nbeats_forecast import NBeats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def calculate_loss(yhat, test_y):
    rmse = mean_squared_error(yhat, test_y) ** 0.5
    r2 = r2_score(test_y, yhat, multioutput = "variance_weighted")
    return rmse, r2


data = pd.read_csv('datasets/weather_energy_hourly.csv')
data = data.avg_energy.values        #univariate time series data of shape nx1 (numpy array)
data = data.reshape(-1,1)
print(data.shape)
model = NBeats(data=data, period_to_forecast=49)
model.fit()
forecast = model.predict()
rmse, r2 = calculate_loss(data[-49:], forecast)
print(rmse, r2)
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print(forecast)
print(data[-49])