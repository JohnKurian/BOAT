import os
import sys
import inspect
import time

from hyperopt import fmin, tpe, hp, Trials
import pickle

from sktime_models import rf_regression, kn_regression, theta_forecaster


# #mlflow stuff
# import mlflow
# # import mlflow.tensorflow
# mlflow.set_tracking_uri("http://sim-mlflow.kn.crc.de.abb.com") # this is optional - if you don't specify, all mlflow runs will be logged locally
# # mlflow.keras.autolog()# this automatically logs data relevant for your model e.g. hyperparameters, evaluation metrics
# mlflow.set_experiment("john_temp_run_sktime")
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['AWS_SECRET_ACCESS_KEY'] = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
# os.environ['AWS_ACCESS_KEY_ID'] = "AKIAIOSFODNN7EXAMPLE"
# os.environ['MLFLOW_S3_ENDPOINT_URL']='http://sim-store.kn.crc.de.abb.com'
#
# #start mlflow
# # run = mlflow.start_run(run_id='dd4a2fcda08649fcbb5b14450c831a60')
# run = mlflow.start_run()
#
# currentdir = os.path.dirname(os.path.abspath(
#     inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)



def run_pipeline(cfg):
    print('#####', cfg)



    cfg = cfg['algo']

    r2 = 0
    rmse = 0


    if cfg['algo'] == 'sk_random_forest':
        r2, rmse = rf_regression(cfg)
    elif cfg['algo'] == 'sk_knn':
        r2, rmse = kn_regression(cfg)
    elif cfg['algo'] == 'theta_forecaster':
        r2, rmse = theta_forecaster(cfg)
    print('here:', r2)
    mlflow.log_metric("r2", r2)
    # mlflow.log_artifact('trials.txt')


    return rmse



def initiate_run():

    space = {
        'algo': hp.choice('algo', [
            {'algo': 'sk_random_forest',
             'window_length': hp.uniformint('sk_random_forest_window_length', 3, 100)
             },
            {'algo': 'sk_knn',
             'neighbours': hp.uniformint('sk_knn_neighbours', 1, 50),
             'window_length': hp.uniformint('sk_knn_window_length', 3, 100)
             },

            {'algo': 'theta_forecaster',
             'sp': hp.uniformint('th_sp', 1, 50)
             }
        ])
    }




    trials = None
    try:
        print('loading from local..')
        trials = pickle.load(open("trials.txt", "rb"))
    except:
        try:
            print('loading from mlflow..')
            from mlflow.tracking import MlflowClient

            # Download artifacts
            client = MlflowClient()
            print(run.info.run_id)
            from mlflow.tracking.artifact_utils import _download_artifact_from_uri

            artifact_path = mlflow.get_artifact_uri() + '/trials.txt'
            print('the artifact path:', artifact_path)
            _download_artifact_from_uri(artifact_path, '.')
            print('success')
        except:
            print('starting a new trial..')
            trials = Trials()
    Trials = None

    best = 0
    if Trials:
        print('here')
        best = fmin(run_pipeline, space, trials=trials, algo=tpe.suggest, max_evals=10000, trials_save_file='trials.txt')
    else:
        best = fmin(run_pipeline, space, trials=trials, algo=tpe.suggest, max_evals=10000)

    mlflow.get_artifact_uri()
    mlflow.end_run()


    return {'time': time.time()}








