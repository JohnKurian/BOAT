
import streamlit as st
import pandas as pd
import numpy as np
import hiplot as hip
import os
from functools import partial
import mlflow

from machine_learning  import rf_regression, kn_regression, theta_forecaster
from lstm import lstm_model, multivariate_lstm_model
# from nbeats_model import nbeats_model
from temporal_convolutional_network  import temporal_convolutional_network

import SessionState

from datetime import datetime






# import graphviz as graphviz
# graph = graphviz.Digraph()
# graph.edge('run', 'intr')
# graph.edge('intr', 'runbl')
# graph.edge('run', 'kernel')
#
# st.graphviz_chart(graph)


# data = [{'dropout':0.1, 'lr': 0.001, 'loss': 10.0, 'optimizer': 'SGD'},
#         {'dropout':0.15, 'lr': 0.01, 'loss': 3.5, 'optimizer': 'Adam'},
#         {'dropout':0.3, 'lr': 0.1, 'loss': 4.5, 'optimizer': 'Adam'}]
#
# hip.Experiment.from_iterable(data).display_st()


def initiate_run(exp_id,
                 dataset_name,
                 predictor,
                 forecasting_horizon,
                 mode,
                 selected_features,
                 selected_algos):

    import math
    from hyperopt import fmin, tpe, hp, Trials
    # from hyperopt.mongoexp import MongoTrials
    import os
    # os.system('hyperopt-mongo-worker --mongo=root:7dc41992@cluster0-shard-00-02.vckj9.mongodb.net:27017/main --poll-interval=0.1 &')
    # os.system('hyperopt-mongo-worker --mongo=localhost:27017/main --poll-interval=0.1 &')

    # import subprocess
    # subprocess.Popen('hyperopt-mongo-worker --mongo=localhost:27017/main --poll-interval=0.1 &', shell=True)

    import pickle

    # global selected_algos, mode

    algo_space_map = {
        'TCN': {'algo': 'tcn',
             'lag_days': hp.uniformint('lag_days', 2, 50)
             },
        'LSTM': {'algo': 'lstm',
             'n_steps_in': hp.uniformint('n_steps_in', 2, 50),
             'hidden_layer_1_neurons': hp.uniformint('hidden_layer_1_neurons', 3, 100),
             'hidden_layer_2_neurons': hp.uniformint('hidden_layer_2_neurons', 3, 100)
             },
        'Sktime-RandomForest': {'algo': 'sk_random_forest',
             'window_length': hp.uniformint('sk_random_forest_window_length', 3, 100)
             },
        'Sktime-KNN': {'algo': 'sk_knn',
             'neighbours': hp.uniformint('sk_knn_neighbours', 1, 50),
             'window_length': hp.uniformint('sk_knn_window_length', 3, 100)
             },
        'Sktime-ThetaForecaster': {'algo': 'theta_forecaster',
             'sp': hp.uniformint('th_sp', 1, 50)
             },
        'NBeats': {'algo': 'nbeats'
             }
    }

    algos = [algo_space_map[selected_algo] for selected_algo in selected_algos]

    space = {
        'algo': hp.choice('algo', algos)
    }


    # trials = None
    # try:
    #     print('loading from local..')
    #     trials = pickle.load(open("trials.txt", "rb"))
    # except:
    #     try:
    #         print('loading from mlflow..')
    #         from mlflow.tracking import MlflowClient
    #
    #         # Download artifacts
    #         client = MlflowClient()
    #         # print(run.info.run_id)
    #         from mlflow.tracking.artifact_utils import _download_artifact_from_uri
    #
    #         # artifact_path = mlflow.get_artifact_uri() + '/trials.txt'
    #         print('the artifact path:', artifact_path)
    #         _download_artifact_from_uri(artifact_path, '.')
    #         print('success')
    #     except:
    #         print('starting a new trial..')
    #         trials = Trials()
    # Trials = None

    best = 0

    data =  {
        'exp_id': exp_id,
        'dataset_name': dataset_name,
        'predictor': predictor,
        'forecasting_horizon': forecasting_horizon,
        'mode': mode,
        'selected_features': selected_features
    }

    # fmin_objective = partial(run_pipeline, chart_data, chart, results, my_slot1, dataset_location, predictor, forecasting_horizon, mode, selected_features)

    def run_mlflow(cfg, data):
        active_run = mlflow.active_run()
        rmse = run_pipeline(cfg, data)
        mlflow.end_run()
        return rmse

    fmin_objective = partial(run_mlflow, data=data)

    # trials = MongoTrials('mongodb://root:7dc41992@cluster0-shard-00-02.vckj9.mongodb.net:27017/main/jobs?retryWrites=true&w=majority', exp_key='trail1')
    # trials = MongoTrials('mongo://localhost:27017/main/jobs', exp_key='trail1')

    import os.path
    import pymongo

    #
    # if 'trials_file_id' in experiment:
    #     trials_file_id = experiment['trials_file_id']
    #
    #     file = open('trials.txt', 'wb+')
    #
    #     fs.download_to_stream(trials_file_id, file)
    #     file.seek(0)
    #     file.close()
    #
    #     print('loading from trials save file..')
    #     trials = pickle.load(open("trials.txt", "rb"))
    #
    # else:
    #     if os.path.isfile('trials.txt'):
    #         os.remove('trials.txt')
    #     trials = Trials()

    trials = Trials()

    best = fmin(fmin_objective, space, trials=trials, algo=tpe.suggest, max_evals=10000, trials_save_file='trials.txt')

    # if Trials:
    #     print('here')
    #     best = fmin(fmin_objective, space, trials=trials, algo=tpe.suggest, max_evals=10000, trials_save_file='trials.txt')
    # else:
    #     best = fmin(fmin_objective, space, trials=trials, algo=tpe.suggest, max_evals=10000)

    # mlflow.get_artifact_uri()
    # mlflow.end_run()


    return 0



best_r2 = -9999

def run_pipeline(cfg, data):
    print('cfg:', cfg)
    print('data:', data)

    exp_id = data['exp_id']
    dataset_name = data['dataset_name']
    predictor = data['predictor']
    forecasting_horizon = data['forecasting_horizon']
    mode = data['mode']
    selected_features = data['selected_features']



    global best_r2
    # global chart_data, chart, results, my_slot1, dataset_name, predictor, forecasting_horizon
    # global mode, selected_features
    print('#####', cfg)

    import gridfs

    # import pymongo
    #
    # client = pymongo.MongoClient(
    #     "mongodb+srv://root:7dc41992@cluster0.vckj9.mongodb.net/main?retryWrites=true&w=majority")
    #
    # db = client.automl
    #
    # fs = gridfs.GridFSBucket(client.automl)
    # # fs.get('5ffe4c4b0bff9f6f25f1b0c1')
    # print(fs.)

    # import os.path
    #
    # if os.path.isfile('trials.txt'):
    #     file_id = fs.upload_from_stream("trials_obj", open(r'trials.txt', 'rb'))
    #     print(file_id)
    #
    #     db.experiments.update({'exp_id': {'$eq': exp_id}},
    #                                      {'$set': {'trials_file_id': file_id}})




    cfg = cfg['algo']

    r2 = 0
    rmse = 0
    y_test = []
    y_pred = []


    if mode == 'univariate':
        if cfg['algo'] == 'sk_random_forest':
            r2, rmse, y_test, y_pred = rf_regression(dataset_name, predictor, forecasting_horizon, cfg, exp_id)
        elif cfg['algo'] == 'sk_knn':
            r2, rmse, y_test, y_pred = kn_regression(dataset_name, predictor, forecasting_horizon, cfg, exp_id)
        elif cfg['algo'] == 'theta_forecaster':
            r2, rmse, y_test, y_pred = theta_forecaster(dataset_name, predictor, forecasting_horizon, cfg, exp_id)
        elif cfg['algo'] == 'lstm':
            r2, rmse, y_test, y_pred = lstm_model(dataset_name, predictor, forecasting_horizon, cfg, exp_id)
        elif cfg['algo'] == 'tcn':
            r2, rmse, y_test, y_pred = temporal_convolutional_network(dataset_name, predictor, forecasting_horizon, cfg, exp_id)
        elif cfg['algo'] == 'nbeats':
            r2, rmse, y_test, y_pred = nbeats_model(dataset_name, predictor, forecasting_horizon, cfg, exp_id)

    else:
        if cfg['algo'] == 'lstm':
            r2, rmse, y_test, y_pred = multivariate_lstm_model(dataset_name, predictor, selected_features, forecasting_horizon, cfg)


    print('here:', r2)



    mlflow.log_metric("r2", r2)


    if cfg['algo'] == 'tcn':
        mlflow.log_metric("model_TCN", 1)
        mlflow.log_metric("model_LSTM", 0)
        mlflow.log_metric("lag_days", cfg['lag_days'])

    elif  cfg['algo'] == 'lstm':
        mlflow.log_metric("model_LSTM", 1)
        mlflow.log_metric("model_TCN", 0)
        mlflow.log_metric("hidden_layer_1_neurons", cfg['hidden_layer_1_neurons'])
        mlflow.log_metric("hidden_layer_2_neurons", cfg['hidden_layer_2_neurons'])
        mlflow.log_metric("n_steps_in", cfg['n_steps_in'])


    # mlflow.log_param("y_test", y_test.tolist())
    # mlflow.log_param("y_pred", y_pred.tolist())
    mlflow.log_artifact('trials.txt')


    # db.runs.insert_one({'exp_id': exp_id,
    #                     'config': cfg,
    #                     'r2': r2,
    #                     'rmse': rmse,
    #                     'y_test': y_test.tolist(),
    #                     'y_pred': y_pred.tolist(),
    #                     'dataset_name': dataset_name,
    #                     'created_at': datetime.utcnow()})

    best_config = None
    # for run in client.automl.runs.find({'exp_id': {'$eq': exp_id}}).sort(
    #         [("r2", pymongo.DESCENDING)]):
    #     best_config = run
    #     break
    #
    # y_test = best_config['y_test']
    # y_pred = best_config['y_pred']

    print(np.vstack((y_test,y_pred)).T.shape)




    return rmse

# uploaded_file = st.file_uploader("Upload the dataset")



def run_ui():


    import uuid, pymongo

    session_state = SessionState.get(choices="",
                                     choose_experiment='',
                                     exp_id=None,
                                     new_exp_name='',
                                     predictor='',
                                     forecasting_horizon='',
                                     selected_features=[],
                                     selected_algos=[],
                                     mode='',
                                     button_sent=False)

    client = pymongo.MongoClient(
        "mongodb+srv://root:7dc41992@cluster0.vckj9.mongodb.net/main?retryWrites=true&w=majority")

    db = client.automl

    choices = ['create new experiment', 'choose from existing experiment']
    session_state.choose_experiment = st.selectbox(
        'Experiment selection',
        choices)


    if session_state.choose_experiment == 'create new experiment':
        new_exp_name = st.text_input("experiment name", 'automl-exp')





        if st.button('create'):
            session_state.exp_id = str(uuid.uuid4())
            db.experiments.insert_one({'exp_id': session_state.exp_id, 'exp_name': new_exp_name, 'created_at': datetime.utcnow()})
            st.text('created exp id {}'.format(session_state.exp_id))


    else:
        exps = list(db.experiments.find({}, {'exp_id': 1, 'exp_name': 1}))

        exp_strings = ['{}({})'.format(exp['exp_name'], exp['exp_id']) for exp in exps]

        exp_map = dict(zip(exp_strings, exps))

        selected_exp_string = st.selectbox(
            'Pick the experiments',
            exp_strings)

        if selected_exp_string:
            selected_exp = exp_map[selected_exp_string]
            session_state.exp_id = selected_exp['exp_id']
            st.text('the seleced experiment is {}'.format(session_state.exp_id))


    import os
    datasets = os.listdir('datasets')
    dataset = st.selectbox(
        'Pick the dataset',
        datasets)
    dataframe = pd.read_csv('datasets/' + dataset)
    dataset_name = 'datasets/' +dataset
    st.write(dataframe)
    st.text('dataset has {} rows'.format(len(dataframe)))
    print(session_state.exp_id)

    dataframe = dataframe.apply(lambda col: pd.to_datetime(col, errors='ignore')
                  if col.dtypes == object
                  else col,
                  axis=0)

    date_cols = [col for col in list(dataframe.columns) if (dataframe[col].dtype == np.dtype('datetime64[ns]'))]
    st.text('Date columns found: {}'.format(date_cols))

    date_index = st.selectbox(
        'Pick the date index',
        date_cols)


    session_state.forecasting_horizon = int(st.text_input("input the forecasting horizon", 49))


    options = list(dataframe.columns)
    session_state.predictor = st.selectbox(
        'Pick the predictor column',
        options)

    st.write('You selected:', session_state.predictor)

    options = ['univariate', 'multivariate']
    session_state.mode = st.selectbox(
        'Select the mode',
        options)

    st.write('You selected:', session_state.mode)



    if session_state.mode == 'multivariate':
        available_algos = ['LSTM']
    else:
        available_algos = ['TCN', 'LSTM', 'Sktime-RandomForest', 'Sktime-KNN', 'Sktime-ThetaForecaster', 'NBeats']


    if session_state.mode == 'multivariate':
        available_features = list(dataframe.columns)
        available_features.remove(session_state.predictor)

        session_state.selected_features = st.multiselect(
            'Selected features',
             available_features,
             available_features)



    selected_algos = st.multiselect(
        'Selected algorithms',
         available_algos,
         ['LSTM'])

    if session_state.predictor:
        st.line_chart(dataframe[session_state.predictor])

    if st.button('initiate automl'):
        results = {
            'best_r2': 0
        }


        chart_data = pd.DataFrame(columns=['r2'])
        chart = st.line_chart()

        my_slot1 = st.empty()

        my_slot2 = st.empty()

        import os
        import pymongo
        from bson.json_util import dumps
        import json

        client = pymongo.MongoClient(
            "mongodb+srv://root:7dc41992@cluster0.vckj9.mongodb.net/main?retryWrites=true&w=majority")

        pipeline = [
            {
                '$match': {'fullDocument.exp_id': session_state.exp_id}
            }]

        runs = []
        r2_scores = []

        for run in client.automl.runs.find({'exp_id': { '$eq': session_state.exp_id } }):
            run = json.loads(dumps(run))
            r2_scores.append(run['r2'])
            runs.append(json.loads(dumps(run)))
        chart.add_rows(r2_scores)

        # print(runs)

        # change_stream = client.automl.runs.watch(pipeline)
        # for change in change_stream:
        #     run = json.loads(dumps(change))['fullDocument']
        #     r2_scores.append(run['r2'])
        #     chart.add_rows([run['r2']])
        #     print(json.loads(dumps(change))['fullDocument'])
        #     print('')  # for readability only

        if session_state.choose_experiment == 'choose from existing experiment':
            pass

            # chart_data = pd.DataFrame(
            #     np.vstack((y_test, y_pred)).T,
            #     columns=['y_test', 'y_pred'])
            # my_slot2.line_chart(chart_data)





        initiate_run(session_state.exp_id, chart_data, chart, results, my_slot1, my_slot2, dataset_name, session_state.predictor, session_state.forecasting_horizon, session_state.mode, session_state.selected_features, selected_algos)





#https://www.thepolyglotdeveloper.com/2019/01/getting-started-mongodb-docker-container-deployment/




