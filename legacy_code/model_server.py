from fastapi import FastAPI
import uvicorn
import numpy as np
from pydantic import BaseModel

#test
#7f8608e2-14c6-4071-864c-9ae5306324fd

#input data [4.6, 11.1, 8.7, 10.0, 11.3, 10.5, 9.9, 11.0, 14.0, 9.2, 9.8, 6.0, 9.8, 9.2, 11.8, 10.3, 7.5, 7.7, 15.8, 14.6, 10.5, 11.3]


class DataInput(BaseModel):
    data:   list = None

app = FastAPI()


def rolling_window(a, window_size):
    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def predict_tcn_model(input, exp_id):
    from tensorflow.keras import Input, Model

    from tcn import TCN, tcn_full_summary
    import pandas as pd

    from numpy import array
    from tensorflow.keras.layers import Dense
    import numpy as np

    import json
    cfg = []
    with open('models/{}/tcn_model/config.json'.format(exp_id)) as f:
        cfg = json.load(f)


    raw_seq = np.array(input)

    min_raw_seq, ptp_raw_seq = cfg['min_raw_seq'], cfg['ptp_raw_seq']

    raw_seq = (raw_seq - min_raw_seq) / ptp_raw_seq

    lag_1 = cfg['lag_days']
    lag_2 = cfg['lag_days']

    forecasting_horizon = cfg['forecasting_horizon']

    n_steps_out = forecasting_horizon

    batch_size = None
    print(cfg['lag_days'])
    i = Input(batch_shape=(batch_size, 1, cfg['lag_days']))

    o = TCN(return_sequences=False)(i)  # The TCN layers are here.
    o = Dense(n_steps_out)(o)

    m = Model(inputs=[i], outputs=[o])
    m.compile(optimizer='adam', loss='mse')
    m.load_weights('models/{}/tcn_model/model_weights.h5'.format(exp_id))

    raw_seq = raw_seq.reshape((1, 1, cfg['lag_days']))
    yhat = m.predict(raw_seq, verbose=0)
    print(yhat)
    return yhat


@app.post("/{exp_id}/best_model/predict")
async def root(exp_id, input: DataInput):
    input_dict = input.dict()
    inp = input_dict['data']
    pred = predict_tcn_model(inp, exp_id).tolist()
    return {"prediction": pred}


if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='127.0.0.1')


#curl -X POST "http://127.0.0.1:8080/7f8608e2-14c6-4071-864c-9ae5306324fd/best_model/predict" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"data\":[4.6,11.1,8.7,10,11.3,10.5,9.9,11,14,9.2,9.8,6,9.8,9.2,11.8,10.3,7.5,7.7,15.8,14.6,10.5,11.3]}"
