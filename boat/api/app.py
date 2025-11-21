from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import os
import json
import pandas as pd
from threading import Thread

from boat.core.engine import initiate_run
from boat.config import Config

app = Flask(__name__)
CORS(app)

# Configure MLFlow
mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
client = MlflowClient()

@app.route('/hello')
def say_hello_world():
    return {'result': "Hello World"}

@app.route('/create_experiment', methods=['POST'])
def create_experiment():
    req_data = request.get_json(force=True)
    experiment_id = mlflow.create_experiment(req_data['exp_name'])
    client.set_experiment_tag(experiment_id, "dataset_location", req_data['dataset_location'])
    client.set_experiment_tag(experiment_id, "notes", req_data['notes'])
    client.set_experiment_tag(experiment_id, "forecasting_horizon", 1) # Defaulting to 1 as per original code?
    client.set_experiment_tag(experiment_id, "mode", req_data['mode'])
    client.set_experiment_tag(experiment_id, "predictor_column", req_data['predictor_column'])
    client.set_experiment_tag(experiment_id, "selected_algos", req_data['selected_algos'])
    return {'experiment_id': experiment_id}

def run_mlflow_parallel(exp_name, experiment_id):
    mlflow.set_experiment(exp_name)
    exp = client.get_experiment(experiment_id)
    dataset_location = 'datasets/' + exp.tags['dataset_location']

    forecasting_horizon = int(exp.tags['forecasting_horizon'])
    mode = exp.tags['mode']
    predictor_column = exp.tags['predictor_column']
    selected_algos = exp.tags['selected_algos']

    import ast
    try:
        selected_algos = ast.literal_eval(selected_algos)
    except:
        pass # Maybe it's already a list or string

    active_run = mlflow.active_run()
    if active_run:
        mlflow.end_run()
    
    initiate_run(
        exp_id=exp_name, # Using exp_name as exp_id for initiate_run? Original code did this.
        dataset_name=dataset_location,
        predictor=predictor_column,
        forecasting_horizon=forecasting_horizon,
        mode=mode,
        selected_features=[predictor_column], # Simplified
        selected_algos=selected_algos
    )

@app.route('/create_run', methods=['POST'])
def create_run():
    experiment_id = request.args.get('experiment_id')
    exp = client.get_experiment(experiment_id)
    exp_name = exp.name

    # Run in background
    thread = Thread(target=run_mlflow_parallel, args=(exp_name, experiment_id))
    thread.start()

    return {'result': "Experiment started"}

@app.route('/get_experiment')
def get_experiment():
    experiment_id = request.args.get('experiment_id')
    exp = client.get_experiment(experiment_id)

    payload = {}
    payload['experiment_id'] = exp.experiment_id
    payload['lifecycle_stage'] = exp.lifecycle_stage
    payload['name'] = exp.name
    payload['dataset_location'] = exp.tags.get('dataset_location')
    payload['forecasting_horizon'] = exp.tags.get('forecasting_horizon')
    payload['mode'] = exp.tags.get('mode')
    payload['notes'] = exp.tags.get('notes')
    payload['predictor_column'] = exp.tags.get('predictor_column')
    payload['selected_algos'] = exp.tags.get('selected_algos')

    # Logic to get best run and all runs...
    # Simplified for now
    return {'payload': payload}

@app.route('/experiments')
def experiments():
    exps = [{'experiment_id': e.experiment_id, 'name': e.name}
            for e in client.list_experiments(view_type=ViewType.ACTIVE_ONLY)]
    return {'result': exps}

@app.route('/datasets')
def get_datasets():
    datasets = os.listdir('datasets')
    return {'datasets': datasets}

@app.route('/datasets/info')
def get_dataset_info():
    data = request.args.get('data')
    df = pd.read_csv('datasets/'+data)
    dataset_columns = list(df.columns)
    return {'dataset_columns': dataset_columns}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
