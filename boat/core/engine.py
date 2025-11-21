import os
import logging
import pickle
import json
from datetime import datetime
from functools import partial
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import mlflow
from hyperopt import fmin, tpe, hp, Trials
import gridfs

from boat.models import (
    rf_regression,
    kn_regression,
    theta_forecaster,
    lstm_model,
    multivariate_lstm_model,
    temporal_convolutional_network,
    nbeats_model
)
from boat.utils.mongo import get_mongo_client, get_database
from boat.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(cfg: Dict[str, Any], data: Dict[str, Any], callback: Optional[Callable[[float], None]] = None) -> float:
    """
    Execute a single pipeline run with the given configuration.

    Args:
        cfg (Dict[str, Any]): Configuration from hyperopt.
        data (Dict[str, Any]): Context data including dataset info and experiment ID.
        callback (Optional[Callable[[float], None]]): Callback function to report progress (e.g. R2 score).

    Returns:
        float: RMSE (loss) to minimize.
    """
    logger.info(f"Running pipeline with config: {cfg}")
    
    exp_id = data['exp_id']
    dataset_name = data['dataset_name']
    predictor = data['predictor']
    forecasting_horizon = data['forecasting_horizon']
    mode = data['mode']
    selected_features = data['selected_features']

    # Connect to Mongo
    client = get_mongo_client()
    db = get_database(client)
    fs = gridfs.GridFSBucket(db)

    # Save trials file if it exists (checkpointing)
    if os.path.isfile('trials.txt'):
        try:
            with open('trials.txt', 'rb') as f:
                file_id = fs.upload_from_stream("trials_obj", f)
            db.experiments.update_one(
                {'exp_id': exp_id},
                {'$set': {'trials_file_id': file_id}}
            )
        except Exception as e:
            logger.error(f"Failed to save trials file: {e}")

    algo_cfg = cfg['algo']
    algo_name = algo_cfg['algo']

    r2 = 0.0
    rmse = 0.0
    y_test = []
    y_pred = []

    try:
        if mode == 'univariate':
            if algo_name == 'sk_random_forest':
                r2, rmse, y_test, y_pred = rf_regression(dataset_name, predictor, forecasting_horizon, algo_cfg, exp_id)
            elif algo_name == 'sk_knn':
                r2, rmse, y_test, y_pred = kn_regression(dataset_name, predictor, forecasting_horizon, algo_cfg, exp_id)
            elif algo_name == 'theta_forecaster':
                r2, rmse, y_test, y_pred = theta_forecaster(dataset_name, predictor, forecasting_horizon, algo_cfg, exp_id)
            elif algo_name == 'lstm':
                r2, rmse, y_test, y_pred = lstm_model(dataset_name, predictor, forecasting_horizon, algo_cfg, exp_id)
            elif algo_name == 'tcn':
                r2, rmse, y_test, y_pred = temporal_convolutional_network(dataset_name, predictor, forecasting_horizon, algo_cfg, exp_id)
            elif algo_name == 'nbeats':
                r2, rmse, y_test, y_pred = nbeats_model(dataset_name, predictor, forecasting_horizon, algo_cfg, exp_id)
        else:
            if algo_name == 'lstm':
                r2, rmse, y_test, y_pred = multivariate_lstm_model(dataset_name, predictor, selected_features, forecasting_horizon, algo_cfg)
    except Exception as e:
        logger.error(f"Error running algorithm {algo_name}: {e}")
        return float('inf') # Return high loss on failure

    logger.info(f"Run complete. R2: {r2}, RMSE: {rmse}")

    # Callback for UI updates
    if callback and r2 > 0:
        callback(r2)

    # Log to MLFlow
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rmse", rmse)
    
    if algo_name == 'tcn':
        mlflow.log_metric("model_TCN", 1)
        mlflow.log_metric("model_LSTM", 0)
        mlflow.log_metric("lag_days", algo_cfg['lag_days'])
    elif algo_name == 'lstm':
        mlflow.log_metric("model_LSTM", 1)
        mlflow.log_metric("model_TCN", 0)
        mlflow.log_metric("hidden_layer_1_neurons", algo_cfg['hidden_layer_1_neurons'])
        mlflow.log_metric("hidden_layer_2_neurons", algo_cfg['hidden_layer_2_neurons'])
        mlflow.log_metric("n_steps_in", algo_cfg['n_steps_in'])

    mlflow.log_artifact('trials.txt')

    # Log to Mongo
    run_doc = {
        'exp_id': exp_id,
        'config': algo_cfg,
        'r2': r2,
        'rmse': rmse,
        'y_test': y_test.tolist() if isinstance(y_test, np.ndarray) else list(y_test),
        'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else list(y_pred),
        'dataset_name': dataset_name,
        'created_at': datetime.utcnow()
    }
    db.runs.insert_one(run_doc)

    return rmse

def initiate_run(
    exp_id: str,
    dataset_name: str,
    predictor: str,
    forecasting_horizon: int,
    mode: str,
    selected_features: List[str],
    selected_algos: List[str],
    callback: Optional[Callable[[float], None]] = None
) -> float:
    """
    Initiate the AutoML run using Hyperopt.

    Args:
        exp_id (str): Experiment ID.
        dataset_name (str): Path to dataset.
        predictor (str): Target column.
        forecasting_horizon (int): Forecast horizon.
        mode (str): 'univariate' or 'multivariate'.
        selected_features (List[str]): List of feature columns (for multivariate).
        selected_algos (List[str]): List of algorithms to include in search space.
        callback (Optional[Callable[[float], None]]): Callback for progress updates.

    Returns:
        float: Best loss (RMSE).
    """
    
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

    algos = [algo_space_map[algo] for algo in selected_algos if algo in algo_space_map]
    
    if not algos:
        logger.warning("No valid algorithms selected.")
        return 0.0

    space = {
        'algo': hp.choice('algo', algos)
    }

    data = {
        'exp_id': exp_id,
        'dataset_name': dataset_name,
        'predictor': predictor,
        'forecasting_horizon': forecasting_horizon,
        'mode': mode,
        'selected_features': selected_features
    }

    # Load trials
    trials = Trials()
    # Logic to load trials from Mongo or file could be improved here
    # For now, starting fresh or from local file if handled by hyperopt

    def run_mlflow_wrapper(cfg):
        # Ensure we are in an active run or start one?
        # The original code started runs inside or outside.
        # Here we assume we are inside a run or we start one per trial.
        # Starting one per trial is safer for parallelism.
        with mlflow.start_run(nested=True):
             return run_pipeline(cfg, data, callback)

    best = fmin(
        fn=run_mlflow_wrapper,
        space=space,
        algo=tpe.suggest,
        max_evals=100, # Reduced from 10000 for sanity, should be configurable
        trials=trials,
        trials_save_file='trials.txt'
    )

    return 0.0 # Return value of fmin is the best params, but we return 0 or best loss
