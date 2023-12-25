# LAMMPS Stream Machine Learning

This is a simple prototype to demonstrate a proof of concept - that we can run LAMMPS
jobs locally, and have them feed into a service (done by way of user space Usernetes) that
will build a model to predict some Y from a set of parameters X. In real world scientific use
cases, the parameters would be meaningful. For my case, I'm going to run LAMMPS at randomly
selected problem sizes and build a simple regression (and we can try other models) to predict
the time. This means I will:

1. First build [django-river-ml](https://github.com/vsoch/django-river-ml) into a container
2. Deploy that container as a service for Kubernetes (or Usernetes) targeted for a model
3. Expose the service with ingress
4. Have a LAMMPS submission script that randomly chooses parameters in some random and submits jobs to the local HPC cluster.
5. The jobs will finish and send their results to add to the training.

At this point we will have an ML server that can serve predictions for a model. We could incrementally use the same API to get the model itself, and show the plot / change over time. We could also do a separate set of runs (test data) and then use that to see how well we did. We can try these out with different models, for fun!

## 1. Machine Learning Server

### Building

Let's first build the Dockerfile.

```bash
docker build -t ghcr.io/converged-computing/lammps-stream-ml:server .
```

And run to see the server starting:

```bash
docker run -it ghcr.io/converged-computing/lammps-stream-ml:server
```
```console
No changes detected
Operations to perform:
  Apply all migrations: admin, auth, authtoken, contenttypes, sessions
Running migrations:
  Applying contenttypes.0001_initial... OK
  Applying auth.0001_initial... OK
  Applying admin.0001_initial... OK
  Applying admin.0002_logentry_remove_auto_add... OK
  Applying admin.0003_logentry_add_action_flag_choices... OK
  Applying contenttypes.0002_remove_content_type_name... OK
  Applying auth.0002_alter_permission_name_max_length... OK
  Applying auth.0003_alter_user_email_max_length... OK
  Applying auth.0004_alter_user_username_opts... OK
  Applying auth.0005_alter_user_last_login_null... OK
  Applying auth.0006_require_contenttypes_0002... OK
  Applying auth.0007_alter_validators_add_error_messages... OK
  Applying auth.0008_alter_user_username_max_length... OK
  Applying auth.0009_alter_user_last_name_max_length... OK
  Applying auth.0010_alter_group_name_max_length... OK
  Applying auth.0011_update_proxy_permissions... OK
  Applying auth.0012_alter_user_first_name_max_length... OK
  Applying authtoken.0001_initial... OK
  Applying authtoken.0002_auto_20160226_1747... OK
  Applying authtoken.0003_tokenproxy... OK
  Applying sessions.0001_initial... OK
Watching for file changes with StatReloader
Performing system checks...

System check identified no issues (0 silenced).
December 25, 2023 - 09:22:19
Django version 5.0, using settings 'app.settings'
Starting development server at http://0.0.0.0:8000/
Quit the server with CONTROL-C.
```

### Running

Let's now test exposing the server to our local machine to interact with. This will eventually go into Kubernetes.

```bash
docker run -it -p 8080:8080 ghcr.io/converged-computing/lammps-stream-ml:server
```

Now let's (interactively) test from Python. I figure since we want to run this alongside lammps, it makes sense to build a second container, and on the cluster we will run with Singularity. 

```bash
docker build -t ghcr.io/converged-computing/lammps-stream-ml:client -f Dockerfile.client .
docker push ghcr.io/converged-computing/lammps-stream-ml:client
singularity pull docker://ghcr.io/converged-computing/lammps-stream-ml:client
```

And then shell in

```bash
singularity shell lammps-stream-ml_client.sif 
```

### Design

Let's open IPython and test interactions. We will have two phases. Some initial setup will:

1. Generate the models of different types

And then each lammps run will:

2. Query the server for the models and submit the data to each

And then as the models are running and updating, another job will be polling the server to get (and save) updated model data. All in all, this will emulate the following steps:

1. Something running on HPC is generating simulation data
2. The output of the simulation is feeding into a machine learning model
3. The output of the model is used for something

For the last point, in a real world thing we would be using it to predict, but for our cases we just want to see how it changes over time (so likely will save the model parameters to plot instead). 

### Testing

Let's first interactively test to figure out which models we want to try. This first snippet would be what
some initial setup script runs (one job to create the empty models).

```python
from river import datasets
from river import linear_model
from river import preprocessing

from riverapi.main import Client

# Connect to the server running here
cli = Client("http://localhost:8080")

# Upload several models to test for lammps - these are different kinds of regressions
regression_model = preprocessing.StandardScaler() | linear_model.LinearRegression(intercept_lr=0.1)

# https://www.geeksforgeeks.org/passive-aggressive-classifiers/
pa_regression = model = linear_model.PARegressor(C=0.01, mode=2, eps=0.1, learn_intercept=False)

# https://riverml.xyz/latest/api/linear-model/BayesianLinearRegression/
bayesian_regression = linear_model.BayesianLinearRegression()

for model in [regression_model, bayesian_regression, pa_regression]:
    model_name = cli.upload_model(model, "regression")
    print("Created model %s" % model_name)
```

At this point we can emulate being a lammps job that is going to do the same, but just send lammps data (one data point)
to each model in the server. We assume the server is deployed just for lammps (but it doesn't have to be).

```python
from riverapi.main import Client

# Connect to the server running here
cli = Client("http://localhost:8080")

# I just made these up - one run would only have one time (Y) and one problem size (x)
lammps_y = [120, 150, 124, 155] 
lammps_x = [
    {"x": 8, "y": 16, "z": 8}, 
    {"x": 8, "y": 8, "z": 16},
    {"x": 8, "y": 14, "z": 8},
    {"x": 8, "y": 9, "z": 17}
]

# This gets the named models
for model_name in cli.models()['models']:
    print(f"Found model {model_name}")
    for i in range(len(lammps_y)):
        y = lammps_y[i]
        x = lammps_x[i]
        print(f"  Using {x} to predict {y}")
        cli.learn(model_name, x=x, y=y)

# Here is how to do a prediction
for model_name in cli.models()['models']:
    print(f"Found model {model_name}")
    for i in range(len(lammps_y)):
        x = lammps_x[i]
        res = cli.predict(model_name, x=x)
        print(res)
```
```console
Found model boopy-parrot
  Using {'x': 8, 'y': 16, 'z': 8} to predict 120
  Using {'x': 8, 'y': 8, 'z': 16} to predict 150
  Using {'x': 8, 'y': 14, 'z': 8} to predict 124
  Using {'x': 8, 'y': 9, 'z': 17} to predict 155
Found model conspicuous-lentil
  Using {'x': 8, 'y': 16, 'z': 8} to predict 120
  Using {'x': 8, 'y': 8, 'z': 16} to predict 150
  Using {'x': 8, 'y': 14, 'z': 8} to predict 124
  Using {'x': 8, 'y': 9, 'z': 17} to predict 155
Found model lovely-carrot
  Using {'x': 8, 'y': 16, 'z': 8} to predict 120
  Using {'x': 8, 'y': 8, 'z': 16} to predict 150
  Using {'x': 8, 'y': 14, 'z': 8} to predict 124
  Using {'x': 8, 'y': 9, 'z': 17} to predict 155
Found model boopy-parrot
{'model': 'boopy-parrot', 'prediction': 123.69271255948297, 'identifier': '541c32ad-bc6b-4020-8131-b0a6edb08455'}
{'model': 'boopy-parrot', 'prediction': 148.851860786418, 'identifier': '1a418466-aa3e-478e-9da8-9a7581904700'}
{'model': 'boopy-parrot', 'prediction': 119.5667010241425, 'identifier': '4835f7cc-ebfc-4862-bc9f-61be297ee6d2'}
{'model': 'boopy-parrot', 'prediction': 156.12276585012535, 'identifier': '23e0afd9-f98e-4e0a-be61-bb2be173c212'}
Found model conspicuous-lentil
{'model': 'conspicuous-lentil', 'prediction': 171.0125077733422, 'identifier': '5413cc9d-bd8a-4f5d-abf8-421e4529b54c'}
{'model': 'conspicuous-lentil', 'prediction': 84.79484638965116, 'identifier': 'b4f0165f-7cfb-44d5-8923-1646f185ea8a'}
{'model': 'conspicuous-lentil', 'prediction': 151.73526597068874, 'identifier': '8c588b6f-156f-4d10-9e72-b0ce1bb2dc81'}
{'model': 'conspicuous-lentil', 'prediction': 93.29488051934324, 'identifier': '360387fa-ee79-40d6-b1d2-d52a488f1be1'}
Found model lovely-carrot
{'model': 'lovely-carrot', 'prediction': 74.5383644905866, 'identifier': '267d0208-b608-4371-a52c-bba1cae63104'}
{'model': 'lovely-carrot', 'prediction': 88.84723402168493, 'identifier': 'b2e3cf13-ac60-4f56-ae7d-43ddb95bc5b4'}
{'model': 'lovely-carrot', 'prediction': 76.53802672891392, 'identifier': 'ee8da467-1a8a-4174-baf8-5d2ba1c83d4f'}
{'model': 'lovely-carrot', 'prediction': 88.63618047474489, 'identifier': '407d48ee-7f9b-441d-b2a0-ef796a0d281a'}
```

And that would be all the job needs to run at the end. Then we would have a polling single job (or just running this ourselves) to get the model from the server as more jobs are submit.

```python
from riverapi.main import Client

# Connect to the server running here
cli = Client("http://localhost:8080")

for model_name in cli.models()['models']:
    print(f"Getting state of {model_name}")
    model_json = cli.get_model_json(model_name)
    print(model_json)
    metrics = cli.metrics(model_name)
    stats = cli.stats(model_name)
```
```console
# This is the model json
{'StandardScaler': {'with_std': True},
 'LinearRegression': {'optimizer': ['SGD',
   {'lr': ['Constant', {'learning_rate': 0.01}]}],
  'loss': ['Squared'],
  'l2': 0.0,
  'l1': 0.0,
  'intercept_init': 0.0,
  'intercept_lr': ['Constant', {'learning_rate': 0.1}],
  'clip_gradient': 1000000000000.0,
  'initializer': ['Zeros']}}

# These are metrics
{'MAE': 96.80479640331933,
 'RMSE': 99.02585314123202,
 'SMAPE': 104.72412027841295}

# These are stats
{'predict': {'n_calls': 4,
  'mean_duration': 851788,
  'mean_duration_human': '851μs788ns',
  'ewm_duration': 881330,
  'ewm_duration_human': '881μs330ns'},
 'learn': {'n_calls': 4,
  'mean_duration': 1950828,
  'mean_duration_human': '1ms950μs828ns',
  'ewm_duration': 1729788,
  'ewm_duration_human': '1ms729μs788ns'}}
```

We could do other stuff too if we want, e.g.,

```python
# Saves to model-name>.pkl in pwd unless you provide a second arg, dest
cli.download_model(model_name)

# Delete the model to cleanup
cli.delete_model(model_name)
```

There are ways to save an identifier and add a label later if we want.

This is going to be cool! I literally just got this idea falling asleep and jumped up to prototype this quick thing, and today I've made progress. This is so simple it just might work!

### TODO

These are next steps I want to do, after I have the automated builds.

- Update settings params to come from environment
- Put into a Kubernetes object, with environment vars from "secrets"
- Try doing the same, exposing via ingress service
- Write small dummy lammps submission script (likely in another container for now) that submits data to API
- Write script to consume API to get model results (and write somewhere for plotting)
