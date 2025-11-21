from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple
import numpy as np

def calculate_metrics(yhat: np.ndarray, test_y: np.ndarray, forecasting_horizon: int) -> Tuple[float, float]:
    """
    Calculate RMSE and R2 score (or explained variance) for predictions.

    Args:
        yhat (np.ndarray): Predicted values.
        test_y (np.ndarray): True values.
        forecasting_horizon (int): The forecasting horizon.

    Returns:
        Tuple[float, float]: A tuple containing (rmse, explained_variance).
    """
    rmse = mean_squared_error(yhat, test_y) ** 0.5
    if forecasting_horizon > 1:
        explained_variance = r2_score(test_y, yhat, multioutput="variance_weighted")
    else:
        # Avoid division by zero if rmse is 0, though unlikely in real data
        explained_variance = 1 / abs(rmse) if rmse != 0 else float('inf')
    return rmse, explained_variance
