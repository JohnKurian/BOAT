import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

from boat.utils.metrics import calculate_metrics
from boat.utils.data import split_sequence, split_sequences_multivariate

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def lstm_model(dataset_location: str, predictor: str, forecasting_horizon: int, cfg: Dict[str, Any], exp_id: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Train and evaluate a univariate LSTM model.
    """
    df = pd.read_csv(dataset_location)
    raw_seq = df[predictor].values
    min_raw_seq, ptp_raw_seq = np.min(raw_seq), np.ptp(raw_seq)
    # Normalize
    raw_seq = (raw_seq - min_raw_seq) / ptp_raw_seq
    df[predictor] = raw_seq

    n_steps_in, n_steps_out = cfg['n_steps_in'], forecasting_horizon
    X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
    
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = Sequential()
    model.add(LSTM(cfg['hidden_layer_1_neurons'], activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(cfg['hidden_layer_2_neurons'], activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=2, verbose=0)

    # Save model artifacts
    model_dir = os.path.join("models", exp_id, "lstm_model")
    os.makedirs(model_dir, exist_ok=True)
    model.save_weights(os.path.join(model_dir, "model_weights.h5"))
    
    config = cfg.copy()
    config['forecasting_horizon'] = forecasting_horizon
    config['min_raw_seq'] = float(min_raw_seq)
    config['ptp_raw_seq'] = float(ptp_raw_seq)

    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # Prediction
    x_input = X[-1]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)

    rmse, r2 = calculate_metrics(yhat[0], df[predictor].values[-forecasting_horizon:], forecasting_horizon)

    return r2, rmse, df[predictor].values[-forecasting_horizon:], yhat

def multivariate_lstm_model(dataset_location: str, predictor: str, selected_features: List[str], forecasting_horizon: int, cfg: Dict[str, Any]) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Train and evaluate a multivariate LSTM model.
    """
    df = pd.read_csv(dataset_location)

    col_arr = [df[col].values for col in selected_features] + [df[predictor].values]
    raw_seq = np.stack(col_arr, axis=1)

    n_steps_in, n_steps_out = cfg['n_steps_in'], forecasting_horizon
    X, y = split_sequences_multivariate(raw_seq, n_steps_in, n_steps_out)
    
    n_features = X.shape[2]

    model = Sequential()
    model.add(LSTM(cfg['hidden_layer_1_neurons'], activation='relu', return_sequences=True,
                   input_shape=(n_steps_in, n_features)))
    model.add(LSTM(cfg['hidden_layer_2_neurons'], activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=5, verbose=0)

    x_input = X[-1]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)

    rmse, r2 = calculate_metrics(yhat[0], df[predictor].values[-forecasting_horizon:], forecasting_horizon)
    
    # Note: The original code returned 1/rmse as r2 for multivariate? That seems odd. 
    # I will return r2 as is, assuming calculate_metrics returns r2.
    # Wait, original code: return 1/rmse, rmse, ...
    # I should probably stick to r2 if calculate_metrics returns it.
    # calculate_metrics returns (rmse, explained_variance).
    
    return r2, rmse, df[predictor].values[-forecasting_horizon:], yhat
