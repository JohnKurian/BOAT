from mlflow import server
import mlflow
import os

os.makedirs(os.path.dirname(os.getcwd() + '\\mlrun_dir'), exist_ok=True)
file_store_path = os.getcwd() + '\\mlrun_dir'
default_artifact_root = os.getcwd() + '\\mlrun_dir'
host = '0.0.0.0'
port = 5000


server. _run_server(
    '.',
    '.',
    host,
    port
)