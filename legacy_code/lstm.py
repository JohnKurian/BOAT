# univariate multi-step vector-output stacked lstm example
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

import pandas as pd

import numpy as np

from metrics import calculate_metrics

# Currently the ‘memory growth’ option should be the same for all GPUs.
# You should set the ‘memory growth’ option before initializing GPUs.

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# split a multivariate sequence into samples
def split_sequences_multivariate(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def lstm_model(dataset_location, predictor, forecasting_horizon, cfg, exp_id):
    df = pd.read_csv(dataset_location)
    raw_seq = list(df[predictor].values)
    min_raw_seq, ptp_raw_seq = np.min(raw_seq), np.ptp(raw_seq)
    raw_seq = (raw_seq - np.min(raw_seq)) / np.ptp(raw_seq)
    df[predictor] = raw_seq

    # choose a number of time steps
    n_steps_in, n_steps_out = cfg['n_steps_in'], forecasting_horizon
    # split into samples
    X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # define model
    model = Sequential()
    model.add(LSTM(cfg['hidden_layer_1_neurons'], activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(cfg['hidden_layer_2_neurons'], activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')

    # # define model
    # model = Sequential()
    # model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
    # model.add(RepeatVector(n_steps_out))
    # model.add(LSTM(100, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(1)))
    # model.compile(optimizer='adam', loss='mse')

    # model = Sequential()
    # model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
    # model.add(Conv1D(filters=128, kernel_size=5, padding='causal', activation='relu', input_shape=[None, 1]))
    # model.add(LSTM(100, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(1)))
    # model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X, y, epochs=2, verbose=0)

    # model.save('models/lstm_model.h5')

    import os

    filename = "models/" + exp_id + "/lstm_model/"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    model.save_weights("models/" + exp_id + "/lstm_model/model_weights.h5")
    print("Saved model to disk")

    import json

    config = cfg
    config['forecasting_horizon'] = forecasting_horizon
    config['min_raw_seq'] = min_raw_seq
    config['ptp_raw_seq'] = ptp_raw_seq

    with open("models/" + exp_id + '/lstm_model/config.json', 'w') as f:
        json.dump(config, f)

    # demonstrate prediction
    x_input = X[-1]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)

    rmse, r2 = calculate_metrics(yhat[0], df[predictor].values[-forecasting_horizon:], forecasting_horizon)

    return r2, rmse, df[predictor].values[-forecasting_horizon:], yhat



def multivariate_lstm_model(dataset_location, predictor, selected_features, forecasting_horizon, cfg):
    df = pd.read_csv(dataset_location)

    cols = list(df.columns)
    cols.remove(predictor)

    col_arr = [df[col].values for col in selected_features] + [df[predictor].values]
    raw_seq = np.stack(col_arr, axis=1)

    # choose a number of time steps
    n_steps_in, n_steps_out = cfg['n_steps_in'], forecasting_horizon
    # split into samples
    X, y = split_sequences_multivariate(raw_seq, n_steps_in, n_steps_out)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = X.shape[2]


    # define model
    model = Sequential()
    model.add(LSTM(cfg['hidden_layer_1_neurons'], activation='relu', return_sequences=True,
                   input_shape=(n_steps_in, n_features)))
    model.add(LSTM(cfg['hidden_layer_2_neurons'], activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')


    # fit model
    model.fit(X, y, epochs=5, verbose=0)

    # demonstrate prediction
    x_input = X[-1]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)

    rmse, r2 = calculate_metrics(yhat[0], df[predictor].values[-forecasting_horizon:], forecasting_horizon)
    print(rmse, r2)
    # x = input('press to continue')

    return 1/rmse, rmse, df[predictor].values[-forecasting_horizon:], yhat

