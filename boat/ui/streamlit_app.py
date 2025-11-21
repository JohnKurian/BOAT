import streamlit as st
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
import os
import json
from bson.json_util import dumps

from boat.core.engine import initiate_run
from boat.utils.mongo import get_mongo_client, get_database
import boat.utils.session_state as SessionState

def run_ui():
    st.title("BOAT: Bayesian Optimization AutoML Time-series")

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

    client = get_mongo_client()
    db = get_database(client)

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

    # Dataset selection
    if not os.path.exists('datasets'):
        st.error("No 'datasets' directory found.")
        return

    datasets = os.listdir('datasets')
    dataset = st.selectbox(
        'Pick the dataset',
        datasets)
    
    if not dataset:
        return

    dataset_path = os.path.join('datasets', dataset)
    dataframe = pd.read_csv(dataset_path)
    dataset_name = dataset_path
    
    st.write(dataframe.head())
    st.text('dataset has {} rows'.format(len(dataframe)))

    # Date column detection
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
        if session_state.predictor in available_features:
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
        chart = st.line_chart()
        my_slot1 = st.empty()
        my_slot2 = st.empty()

        # Display existing runs
        r2_scores = []
        for run in client.automl.runs.find({'exp_id': { '$eq': session_state.exp_id } }):
            run = json.loads(dumps(run))
            r2_scores.append(run.get('r2', 0))
        
        if r2_scores:
            chart.add_rows(r2_scores)

        def ui_callback(r2_val):
            chart.add_rows([r2_val])

        initiate_run(
            exp_id=session_state.exp_id,
            dataset_name=dataset_name,
            predictor=session_state.predictor,
            forecasting_horizon=session_state.forecasting_horizon,
            mode=session_state.mode,
            selected_features=getattr(session_state, 'selected_features', []),
            selected_algos=selected_algos,
            callback=ui_callback
        )

if __name__ == '__main__':
    run_ui()
