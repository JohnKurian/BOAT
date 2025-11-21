import os
from random import random, randint
import mlflow
from mlflow import log_metric, log_param, log_artifacts
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
# run = mlflow.start_run(run_id='dd4a2fcda08649fcbb5b14450c831a60')
mlclient = mlflow.tracking.MlflowClient()
mlflow.set_tracking_uri("http://localhost:5000")
# experiment = mlflow.create_experiment('test_experiment_2')
# mlflow.set_experiment('test_experiment_3')

# mlflow.set_experiment('0')

# run = mlflow.start_run()
#
# tags = {"engineering": "ML Platform",
#         "release.candidate": "RC1",
#         "release.version": "2.2.0"}

mlclient.set_experiment_tag('0',
     "mlflow.note.content","this is a custom note")

experiment = mlclient.get_experiment('0')
print("Name: {}".format(experiment.name))
print("Tags: {}".format(experiment.tags))

# mlflow.set_tags(tags)
#
# print(run.info.run_id)


# if __name__ == "__main__":
#     # Log a parameter (key-value pair)
#     log_param("param1", randint(0, 100))
#
#     # Log a metric; metrics can be updated throughout the run
#     log_metric("foo", random())
#     log_metric("foo", random() + 1)
#     log_metric("foo", random() + 2)
#
#     # Log an artifact (output file)
#     if not os.path.exists("outputs"):
#         os.makedirs("outputs")
#     with open("outputs/new_test.txt", "w") as f:
#         f.write("hello world!")
#     log_artifacts("outputs")
#     mlflow.end_run()
