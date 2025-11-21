import mlflow

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://localhost:5000")
mlclient = MlflowClient()


runs = mlflow.search_runs(experiment_ids='30')
cols = list(runs.columns)
print(cols)
selected_cols = ['start_time', 'metrics.hidden_layer_1_neurons', 'metrics.n_steps_in', 'metrics.model_TCN', 'metrics.model_LSTM', 'metrics.hidden_layer_2_neurons', 'metrics.r2', 'metrics.lag_days']
cols = selected_cols
runs = runs[cols]
exit()
key_list = list(range(len(runs)))
runs['key'] = key_list
cols_json = [{'title': col, 'dataIndex': col, 'key': col} for col in cols]
data_source = runs.to_json(orient="records")

print(data_source)


exit()

all_runs = MlflowClient().search_runs(
        experiment_ids=['28'],
        filter_string="",
        run_view_type=ViewType.ALL
    )


# all_runs = [run.data.metrics['r2'] for run in all_runs]
print(all_runs)

exit()

best_run = mlclient.search_runs(
  experiment_ids='28',
  filter_string="",
  run_view_type=ViewType.ALL,
  max_results=1,
  order_by=["metrics.r2 DESC"]
)[0]

print(best_run.data.metrics)

exit()

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


run_ids = [run.run_id for run in mlclient.list_run_infos('27')]



r2_values = print_metric_info(mlclient.get_metric_history(run_ids[0], 'model_TCN'))


print(r2_values)

exit()

# print(mlclient.get_run('c526b082f55e488fab7234b55ab561d0'))

# new_experiment = mlflow.create_experiment('blah blah 1')
# print(new_experiment)

print(MlflowClient().search_runs(
        experiment_ids=['27'],
        filter_string="",
        run_view_type=ViewType.ALL
    ))
print('done')
exit()


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



run_ids = [run.run_id for run in mlclient.list_run_infos('27')]

for run_id in run_ids:
    r2_values = print_metric_info(mlclient.get_metric_history(run_id, 'r2'))

exit()

experiment_id = '27'

all_runs_search = MlflowClient().search_runs(
  experiment_ids=experiment_id,
  filter_string="",
  run_view_type=ViewType.ALL
)


for run in all_runs_search:
    print(run)


all_r2 = [run.data.metrics['r2'] for run in all_runs_search]
print(all_r2)
exit()


exp = mlclient.get_experiment('25')
print(exp.experiment_id)
print(exp.lifecycle_stage)
print(exp.name)
print(exp.tags['dataset_location'])
print(exp.tags['forecasting_horizon'])
print(exp.tags['mode'])
print(exp.tags['notes'])
print(exp.tags['predictor_column'])
print(exp.tags['selected_algos'])

best_run = MlflowClient().search_runs(
  experiment_ids=exp.experiment_id,
  filter_string="",
  run_view_type=ViewType.ALL,
  max_results=1,
  order_by=["metrics.r2 DESC"]
)[0]


print(best_run)

if best_run.data.metrics['model_LSTM'] == 1:
    print(best_run.data.metrics['hidden_layer_1_neurons'])
    print(best_run.data.metrics['hidden_layer_2_neurons'])
    print(best_run.data.metrics['n_steps_in'])

elif best_run.data.metrics['model_TCN'] == 1:
    print(best_run.data.metrics['lag_days'])

print(best_run.data.metrics['r2'])
print(best_run.info.run_id)

exit()

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType






run = MlflowClient().search_runs(
  experiment_ids="1",
  filter_string="",
  run_view_type=ViewType.ALL,
  max_results=1,
  order_by=["metrics.r2 DESC"]
)[0]

# print(run)



exit()





def print_experiment_info(experiments):
    for e in experiments:
        print("- experiment_id: {}, name: {}, lifecycle_stage: {}"
              .format(e.experiment_id, e.name, e.lifecycle_stage))

client = MlflowClient()
# for name in ["Experiment 1", "Experiment 2"]:
#     exp_id = client.create_experiment(name)

# # Delete the last experiment
# client.delete_experiment(exp_id)

# Fetch experiments by view type
print("Active experiments:")
print_experiment_info(client.list_experiments(view_type=ViewType.ACTIVE_ONLY))
print("Deleted experiments:")
print_experiment_info(client.list_experiments(view_type=ViewType.DELETED_ONLY))
print("All experiments:")
print_experiment_info(client.list_experiments(view_type=ViewType.ALL))
print('here')

import pdb


# get runs for a particular experiment id

for run in client.list_run_infos('0'):
    print(run.start_time, run.end_time)
    print(run.run_id)
    print(run.artifact_uri)


    # get metrics params and tags for a run
    print('hey there here')

    run_obj = client.get_run(run.run_id)

    print(client.get_run(run.run_id))
    print(client.list_artifacts(run.run_id))


#https://www.mlflow.org/docs/latest/search-syntax.html

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

run = MlflowClient().search_runs(
  experiment_ids="0",
  filter_string="",
  run_view_type=ViewType.ACTIVE_ONLY,
  max_results=1,
  order_by=["metrics.foo DESC"]
)[0]

print(run)



from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

query = "params.model = 'CNN' and params.layers = '10' and metrics.`prediction accuracy` >= 0.945"
runs = MlflowClient().search_runs(
    experiment_ids=["3", "4"],
    filter_string=query,
    run_view_type=ViewType.ACTIVE_ONLY
)





from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

all_experiments = [exp.experiment_id for exp in MlflowClient().list_experiments()]
runs = MlflowClient().search_runs(
    experiment_ids=all_experiments,
    filter_string="params.model = 'Inception'",
    run_view_type=ViewType.ALL
)



