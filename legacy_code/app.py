from flask import Flask
from flask_cors import CORS
import mlflow

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from flask import request



import streamlit as st
import pandas as pd
import numpy as np
import hiplot as hip
import os
from functools import partial

from machine_learning  import rf_regression, kn_regression, theta_forecaster
from lstm import lstm_model, multivariate_lstm_model
# from nbeats_model import nbeats_model
from temporal_convolutional_network  import temporal_convolutional_network

import SessionState

from datetime import datetime



from automl import initiate_run

mlflow.set_tracking_uri("http://localhost:5000")


client = MlflowClient()

# mlflow.tensorflow.save_model(model, 'tft_model')





app = Flask(__name__)
CORS(app)


def print_metric_info(history):
    r2_values = []
    for m in history:
        print("name: {}".format(m.key))
        print("value: {}".format(m.value))
        print("step: {}".format(m.step))
        print("timestamp: {}".format(m.timestamp))
        print("--")
        r2_values.append(m.value)
    return r2_values



@app.route('/hello')
def say_hello_world():
    return {'result': "Hello World"}



@app.route('/create_experiment', methods=['POST'])
def create_experiment():
    print('here')
    req_data = request.get_json(force=True)
    print(req_data)
    experiment = mlflow.create_experiment(req_data['exp_name'])
    client.set_experiment_tag(experiment,"dataset_location", req_data['dataset_location'])
    client.set_experiment_tag(experiment,"notes", req_data['notes'])
    # client.set_experiment_tag(experiment, "forecasting_horizon", req_data['forecasting_horizon'])
    client.set_experiment_tag(experiment, "forecasting_horizon", 1)
    client.set_experiment_tag(experiment, "mode", req_data['mode'])
    client.set_experiment_tag(experiment, "predictor_column", req_data['predictor_column'])
    client.set_experiment_tag(experiment, "selected_algos", req_data['selected_algos'])
    return {'experiment_id': experiment}


@app.route('/start_run')
def start_run():
    client.get_run('c526b082f55e488fab7234b55ab561d0')
    return {'result': "Hello experiment"}


from threading import Thread
from time import sleep

def threaded_function(arg):
    for i in range(arg):
        print("running")
        sleep(1)

def run_mlflow_parallel(exp_name, experiment_id):
    # do something
    mlflow.set_experiment(exp_name)
    exp = client.get_experiment(experiment_id)
    dataset_location = 'datasets/' + exp.tags['dataset_location']

    forecasting_horizon = int(exp.tags['forecasting_horizon'])
    mode = exp.tags['mode']
    predictor_column = exp.tags['predictor_column']
    selected_algos = exp.tags['selected_algos']

    import ast
    selected_algos = ast.literal_eval(selected_algos)

    active_run = mlflow.active_run()
    if active_run is None:
        # with mlflow.start_run() as run:
        initiate_run(exp_name,
                     dataset_location,
                     predictor_column,
                    forecasting_horizon,
                     mode,
                     [predictor_column],
                     selected_algos)
    else:
        print('ending previous run')
        mlflow.end_run()
        print('starting new run')
        # with mlflow.start_run() as run:
        initiate_run(exp_name,
                     dataset_location,
                     predictor_column,
                    forecasting_horizon,
                     mode,
                     [predictor_column],
                     selected_algos)




from multiprocessing.pool import ThreadPool

@app.route('/create_run', methods=['POST'])
def create_run():
    experiment_id = request.args.get('experiment_id')

    exp = client.get_experiment(experiment_id)
    exp_name = exp.name

    run_mlflow_parallel(exp_name, experiment_id)
    # thread = Thread(target = run_mlflow_parallel, args = (exp_name, experiment_id ))
    # thread.start()

    # pool = ThreadPool(processes=100)
    # pool.map(run_mlflow_parallel, [exp_name])


    return {'result': "Hello experiment"}


@app.route('/get_experiment')
def get_experiment():
    experiment_id = request.args.get('experiment_id')
    exp = client.get_experiment(experiment_id)

    payload = {}

    payload['experiment_id'] = exp.experiment_id
    payload['lifecycle_stage'] = exp.lifecycle_stage
    payload['name'] = exp.name
    payload['dataset_location'] = exp.tags['dataset_location']
    payload['forecasting_horizon'] = exp.tags['forecasting_horizon']
    payload['mode'] = exp.tags['mode']
    payload['notes'] = exp.tags['notes']
    payload['predictor_column'] = exp.tags['predictor_column']
    payload['selected_algos'] = exp.tags['selected_algos']

    best_run = MlflowClient().search_runs(
        experiment_ids=exp.experiment_id,
        filter_string="",
        run_view_type=ViewType.ALL,
        max_results=1,
        order_by=["metrics.r2 DESC"]
    )

    if len(best_run) > 0:

        all_runs = MlflowClient().search_runs(
            experiment_ids=['28'],
            filter_string="",
            run_view_type=ViewType.ALL
        )

        all_r2 = [run.data.metrics['r2'] for run in all_runs]
        payload['all_r2'] = all_r2

        # best_run = best_run[0]
        # try:
        #     if best_run.data.metrics['model_LSTM'] == 1:
        #         payload['model'] = 'LSTM'
        #         payload['hidden_layer_1_neurons'] = best_run.data.metrics['hidden_layer_1_neurons']
        #         payload['hidden_layer_2_neurons'] = best_run.data.metrics['hidden_layer_2_neurons']
        #         payload['n_steps_in'] = best_run.data.metrics['n_steps_in']
        #
        #
        #     elif best_run.data.metrics['model_TCN'] == 1:
        #         payload['model'] = 'TCN'
        #         payload['lag_days'] = best_run.data.metrics['lag_days']
        # except KeyError:
        #     pass
        #
        # payload['r2'] = best_run.data.metrics['r2']
        # payload['best_run_id'] = best_run.info.run_id
        #
        # runs = mlflow.search_runs(experiment_ids=experiment_id)
        # cols = list(runs.columns)
        # selected_cols = ['metrics.hidden_layer_1_neurons', 'metrics.n_steps_in', 'metrics.model_TCN',
        #                  'metrics.model_LSTM', 'metrics.hidden_layer_2_neurons', 'metrics.r2', 'metrics.lag_days']
        # cols = selected_cols
        # runs = runs[cols]
        #
        # runs.columns = ['hl_1_neurons', 'n_steps_in', 'TCN',
        #                  'LSTM', 'hl_2_neurons', 'r2', 'lag_days']
        # cols = ['hl_1_neurons', 'n_steps_in', 'TCN',
        #                  'LSTM', 'hl_2_neurons', 'r2', 'lag_days']
        # key_list = list(range(1,len(runs)+1))
        # runs['key'] = key_list
        # cols_json = [{'title': col, 'dataIndex': col, 'key': col} for col in cols]
        # data_source = runs.to_json(orient="records")
        #
        # payload['data_columns'] = cols_json
        #
        # import json
        #
        # payload['datasource'] = json.loads(data_source)






    return {'payload': payload}



@app.route('/get_runs')
def get_runs():
    experiment_id = request.args.get('experiment_id')

    runs = [{'run_id': run.run_id} for run in client.list_run_infos(experiment_id)]

    return {'result': runs}





@app.route('/datasets')
def get_datasets():
    datasets = os.listdir('datasets')
    return {'datasets': datasets}

import pandas as pd

@app.route('/datasets/info')
def get_dataset_info():
    data = request.args.get('data')
    df = pd.read_csv('datasets/'+data)

    dataset_columns = list(df.columns)

    return {'dataset_columns': dataset_columns}


@app.route('/dataset/upload', methods = ['POST'])
def upload_file():
    file = request.files['file']
    print(file)
    return "done"


@app.route('/runs')
def get_run():
    run_id = request.args.get('run_id')
    run_obj = client.get_run(run_id)
    run = [{'metrics': run_obj.data.metrics,
           'params': run_obj.data.params,
           'tags': run_obj.data.tags
           }]

    print(run)
    return {'result': run}








@app.route('/experiments')
def experiments():
    exps = [{'experiment_id': e.experiment_id,
             'name': e.name}
            for e in client.list_experiments(view_type=ViewType.ACTIVE_ONLY)]

    return {'result': exps}