from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



def calculate_metrics(yhat, test_y, forecasting_horizon):
    rmse = mean_squared_error(yhat, test_y) ** 0.5
    if forecasting_horizon > 1:
        explained_variance = r2_score(test_y, yhat, multioutput = "variance_weighted")
    else:
        explained_variance = 1/abs(rmse)
    return rmse, explained_variance