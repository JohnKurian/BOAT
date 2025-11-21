import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tcn import TCN, tcn_full_summary

from boat.utils.metrics import calculate_metrics
from boat.utils.data import split_sequence

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def temporal_convolutional_network(dataset_location: str, predictor: str, forecasting_horizon: int, cfg: Dict[str, Any], exp_id: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Train and evaluate a TCN model.
    """
    df = pd.read_csv(dataset_location)
    raw_seq = df[predictor].values
    min_raw_seq, ptp_raw_seq = np.min(raw_seq), np.ptp(raw_seq)
    # Normalize
    raw_seq = (raw_seq - min_raw_seq) / ptp_raw_seq
    df[predictor] = raw_seq

    # choose a number of time steps
    n_steps_in = cfg['lag_days']
    n_steps_out = forecasting_horizon
    
    # split into samples
    x, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
    
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    x = x.reshape((x.shape[0], x.shape[1], n_features))

    batch_size = None
    i = Input(batch_shape=(batch_size, n_steps_in, n_features))

    o = TCN(return_sequences=False)(i)
    o = Dense(n_steps_out)(o)

    m = Model(inputs=[i], outputs=[o])
    m.compile(optimizer='adam', loss='mse')

    # tcn_full_summary(m, expand_residual_blocks=False) # Optional logging

    m.fit(x, y, epochs=100, validation_split=0.2, shuffle=False, verbose=0)

    # Save model artifacts
    model_dir = os.path.join("models", exp_id, "tcn_model")
    os.makedirs(model_dir, exist_ok=True)
    m.save_weights(os.path.join(model_dir, "model_weights.h5"))
    
    config = cfg.copy()
    config['forecasting_horizon'] = forecasting_horizon
    config['min_raw_seq'] = float(min_raw_seq)
    config['ptp_raw_seq'] = float(ptp_raw_seq)

    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # Prediction
    x_input = x[-1]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = m.predict(x_input, verbose=0)

    rmse, r2 = calculate_metrics(yhat[0], df[predictor].values[-forecasting_horizon:], forecasting_horizon)

    return r2, rmse, df[predictor].values[-forecasting_horizon:], yhat
