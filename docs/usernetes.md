# Usernetes

We are going to try and reproduce the setup in [kubernetes](kubernetes.md) but with usernetes on a private cluster, and running lammps via submit jobs instead of otherwise. The details of the cluster are left out here, but you can assume you need to ssh into it and have dual access to usernetes and flux. Before starting these steps you should have your usernetes cluster running alongside flux, and each of `flux resource list` and `kubectl get nodes` should be working.

## Setup

### Singularity

We will be running the client via a singularity container.
You can create a directory for your running context.

```bash
flux exec mkdir -p ~/lammps-ml
```

From a flux instance, pull the singularity container to each cluster node. This is what we will use to generate
the models and run lammps.

```bash
flux exec singularity pull --dir /home/flux/lammps-ml docker://ghcr.io/converged-computing/lammps-stream-ml:lammps
```

While ideally we could run everything in the container, we are going to be a bit lazy and install our client on the host, and that way we don't need to write complex scripts to interact with a container multiple times.

```bash
flux exec pip3 install river riverapi
```

In the future a cool solution would be having the ability to interact with Flux from the container.

### Usernetes

Note that usernetes has been setup to allow unprivileged ports:

```bash
echo net.ipv4.ip_unprivileged_port_start=0 >> /etc/sysctl.conf 
sysctl -p
systemctl daemon-reload
```

No further action required on your part (unless you didn't do this)! Enable autocomplete to make life easier:

```bash
source <(kubectl completion bash)
```

Let's get the files that we need. These only need to be downloaded to the control plane.

```bash
cd ./lammps-ml
wget https://raw.githubusercontent.com/converged-computing/lammps-stream-ml/main/k8s/server-deployment.yaml
```

### Ingress

Note - I tested this without ingress (see [kubernetes](kubernetes.md) and confirmed that it worked, so I didn't add it. In the future with fully working usernetes we will need it.

### Deployment

Deploy the machine learning server.

```bash
kubectl apply -f ./server-deployment.yaml
```

Note that we have hard coded secrets, which is OK for local testing, but you should update these to a secret proper for anything more than that. At this point, we need to find the node that the server is running on:

```bash
kubectl  get pods -o wide --watch
```
```console
NAME                         READY   STATUS    RESTARTS   AGE   IP           NODE           NOMINATED NODE   READINESS GATES
ml-server-657b4ccf6c-6sdk6   1/1     Running   0          32s   10.244.4.2   u7s-u2204-05   <none>           <none>
```

And then we can ping the API at port 8080.

```bash
$ curl -sk u2204-05:8080/api/ | jq
```
```console
{
  "id": "django_river_ml",
  "status": "running",
  "name": "Django River ML Endpoint",
  "description": "This service provides an api for models",
  "documentationUrl": "https://vsoch.github.io/django-river-ml",
  "storage": "shelve",
  "river_version": "0.21.0",
  "version": "0.0.21"
}
```

### Create Models

Let's create three empty models. Since we know the service is running, we can now use our lammps container. It should be in the present working directory:

```bash
flux@u2204-01:~/lammps-ml$ ls
lammps-stream-ml_lammps.sif  server-deployment.yaml
```

Here is how to create the models for the running server. The names will be funny but largely don't matter - we can get them programmatically later. We do need to tell the script the hostname the server is running at:

```bash
singularity exec lammps-stream-ml_lammps.sif python3 /code/1-create-models.py http://u2204-05:8080/
```
```console
Preparing to create models for client URL http://u2204-05:8080/
Created model sticky-frito  # linear regression
Created model loopy-fudge   # pa regression
Created model swampy-cherry # bayesian linear regression
```

### Train LAMMPS

Now let's run our script that is going to run LAMMPS (via flux run) and send the results to the server to train. This requires a different setup than our initial testing because we need the script to submit the flux jobs and target the container, and then (using the `riverapi` installed to the host) upload a training result. The difference here is that since we are calling to flux, this script is run directly on the host. Let's download it first:

```bash
wget https://raw.githubusercontent.com/converged-computing/lammps-stream-ml/main/scripts/2-run-lammps-flux.py
```

You'll notice two actions - to train or predict:

```bash
$ python3 2-run-lammps-flux.py --help
```

<details>

<summary> Flux Run Script Output </summary>

```console
LAMMPS Run (Train or Test) Flux

positional arguments:
  url              URL where ml-server is deployed

options:
  -h, --help       show this help message and exit

actions:
  actions

  {train,predict}  actions
```

</details>

Let's look at train:

```bash
$ python3 2-run-lammps-flux.py train --help
```

<details>

<summary> Flux Run Train Output </summary>

```console
options:
  -h, --help            show this help message and exit
  --workdir WORKDIR     Working directory to run lammps from.
  --container CONTAINER
                        Path to container to run with lammps
  --in INPUTS           Input and parameters for lammps
  --nodes NODES         number of nodes (N)
  --log LOG             write log to path (keep in mind Singularity container
                        is read only)
  --np NP               number of processes per node
  --x-min X_MIN         min dimension for x
  --x-max X_MAX         max dimension for x
  --y-min Y_MIN         min dimension for y
  --y-max Y_MAX         max dimension for y
  --z-min Z_MIN         min dimension for z
  --z-max Z_MAX         max dimension for z
  --iters ITERS         iterations to run of lammps
```

</details>

Now let's train, allowing each parameter to range between these values that we previously timed (the script controls this)

 - x: 1 to 32
 - y: 1 to 8
 - z: 1 to 16

This is a bit arbitrary based on my manual testing, but I had to cap the max runtime. with some strategy. 

#### Test Train Run

Let's do a test run first (one iteration, and flux run so we see everything).

```bash
container=/home/flux/lammps-ml/lammps-stream-ml_lammps.sif
python3 2-run-lammps-flux.py train --container $container --np 48 --nodes 6 --workdir /opt/lammps/examples/reaxff/HNS --x-min 1 --x-max 32 --y-min 1 --y-max 8 --z-min 1 --z-max 16 --iters 1 --url http://u2204-05:8080/
```

The script will ping the server first to ensure that you got the right address, and then run your N iterations.

```console
Preparing to run lammps and train models with /home/flux/lammps-ml/lammps-stream-ml_lammps.sif
{
    "id": "django_river_ml",
    "status": "running",
    "name": "Django River ML Endpoint",
    "description": "This service provides an api for models",
    "documentationUrl": "https://vsoch.github.io/django-river-ml",
    "storage": "shelve",
    "river_version": "0.21.0",
    "version": "0.0.21"
}

🎄️ Running iteration 0 with chosen x: 20 y: 4 z: 7
         flux => /usr/bin/flux run -N 6 --ntasks 48 -c 1 -o cpu-affinity=per-task
  singularity => /usr/bin/singularity exec --pwd /opt/lammps/examples/reaxff/HNS /home/flux/lammps-ml/lammps-stream-ml_lammps.sif /usr/bin/lmp -v x 20 -v y 4 -v z 7 -log /tmp/lammps.log -in in.reaxc.hns -nocite
       result => Lammps run took 41 seconds
Preparing to send LAMMPS data to http://u2204-05:8080/
  Training loopy-fudge with {'x': 20, 'y': 4, 'z': 7} to predict 41
  Training sticky-frito with {'x': 20, 'y': 4, 'z': 7} to predict 41
  Training swampy-cherry with {'x': 20, 'y': 4, 'z': 7} to predict 41
```

#### Train Runs

Now we can run for more iterations. Note that if you are afraid of your terminal quitting you can use `screen` first.

```bash
container=/home/flux/lammps-ml/lammps-stream-ml_lammps.sif
python3 2-run-lammps-flux.py train --container $container --np 48 --nodes 6 --workdir /opt/lammps/examples/reaxff/HNS --x-min 1 --x-max 32 --y-min 1 --y-max 8 --z-min 1 --z-max 16 --iters 20 --url http://u2204-05:8080/
```

You can watch the output to see chosen parameters and output (as shown in the example). This will run your lammps to generate training data, and send it to the server, training each (of three) models.  Note that I chose to (in total) do 1000 training points.


#### Checking Intermediate Status

Note that from a different console you can monitor progress. At this point, we won't see any predictions, but we will see changing stats and the learn endpoint being called! The client we installed locally easy functions to do this:

```python
from riverapi.main import Client

cli = Client('http://u2204-05:8080')

for model_name in cli.models()['models']:
    print("  => Model:")
    print(cli.get_model_json(model_name))
    print("  => Metrics:")
    print(cli.metrics(model_name))
    print("  => Stats:")
    print(cli.stats(model_name))
```

If you need to do this without the client, here are raw examples:

<details>

<summary>River API RESTFul Example</summary>

```python
res = requests.get('http://u2204-04:8080/api/metrics/', json={'model': 'rainbow-kerfuffle'})
```
```console
<Response [200]>
```
```python
res.json()
```
```console
{'MAE': 3.065755887386161,
 'RMSE': 4.075563797945497,
 'SMAPE': 30.025394464628064}
```
```python
res = requests.get('http://u2204-04:8080/api/stats/', json={'model': 'rainbow-kerfuffle'})
res.json()
```
```console
{'predict': {'n_calls': 0,
  'mean_duration': 0,
  'mean_duration_human': '0ns',
  'ewm_duration': 0,
  'ewm_duration_human': '0ns'},
 'learn': {'n_calls': 24,
  'mean_duration': 887652,
  'mean_duration_human': '887μs652ns',
  'ewm_duration': 846600,
  'ewm_duration_human': '846μs600ns'}}
```

</details>

After running this a bunch of times (how many I'm not sure, maybe 100?) We will use predict on it to see how well our models do. This is where we will calculate error metrics based on predictions and actual values. 

### Predict LAMMPS

This will allow us to generate test data on the fly, and then calculate metrics for some number of predictions.

#### Test Prediction Run

Let's start again wtih testing. The reason that I combined these previously two scripts (e.g., used in [kubernetes](kubernetes.md)) is because there is a lot of shared logic to run lammps. Arguably the same should be done for that script, and I'll do this if/when I go back to it.

```bash
container=/home/flux/lammps-ml/lammps-stream-ml_lammps.sif
python3 2-run-lammps-flux.py predict --container $container --np 48 --nodes 6 --workdir /opt/lammps/examples/reaxff/HNS --x-min 1 --x-max 32 --y-min 1 --y-max 8 --z-min 1 --z-max 16 --iters 3 --url http://u2204-05:8080/ --out test-predict.json
```

The above will save the actual values, predictions, metadata about models and error values calculated (metrics).

<details>

<summary> Example Output for Predict</summary>

```console
Preparing to run lammps and predict models with /home/flux/lammps-ml/lammps-stream-ml_lammps.sif
{
    "id": "django_river_ml",
    "status": "running",
    "name": "Django River ML Endpoint",
    "description": "This service provides an api for models",
    "documentationUrl": "https://vsoch.github.io/django-river-ml",
    "storage": "shelve",
    "river_version": "0.21.0",
    "version": "0.0.21"
}

🎄️ Running iteration 0 with chosen x: 6 y: 8 z: 14
         flux => /usr/bin/flux run -N 6 --ntasks 48 -c 1 -o cpu-affinity=per-task
  singularity => /usr/bin/singularity exec --pwd /opt/lammps/examples/reaxff/HNS /home/flux/lammps-ml/lammps-stream-ml_lammps.sif /usr/bin/lmp -v x 6 -v y 8 -v z 14 -log /tmp/lammps.log -in in.reaxc.hns -nocite
       result => Lammps run took 52 seconds
Model loopy-fudge predicts 52.69838420107794
Model sticky-frito predicts 21.829333333333334
Model swampy-cherry predicts 54.5520722729006

⭐️ Performance for: loopy-fudge
          R Squared Error: 0.0
       Mean Squared Error: 0.4877404923152762
      Mean Absolute Error: 0.6983842010779426
  Root Mean Squared Error: 0.6983842010779426

⭐️ Performance for: sticky-frito
          R Squared Error: 0.0
       Mean Squared Error: 910.2691271111111
      Mean Absolute Error: 30.170666666666666
  Root Mean Squared Error: 30.170666666666666

⭐️ Performance for: swampy-cherry
          R Squared Error: 0.0
       Mean Squared Error: 6.513072886108047
      Mean Absolute Error: 2.5520722729006025
  Root Mean Squared Error: 2.5520722729006025
```

</details>

And then the same can be done for a larger number of test cases.

#### Predictions

We can run a test with some number of iterations (new test cases of LAMMPS). Note that we add an `--out` json file to save results to. I decided to do a total of 300 test cases, to give an approximate total of 1250 cases, 80% for training (1000) and 20% for testing (250).

```bash
container=/home/flux/lammps-ml/lammps-stream-ml_lammps.sif
python3 2-run-lammps-flux.py predict --container $container --np 48 --nodes 6 --workdir /opt/lammps/examples/reaxff/HNS --x-min 1 --x-max 32 --y-min 1 --y-max 8 --z-min 1 --z-max 16 --iters 250 --url http://u2204-05:8080/ --out lammps-predict.json
```

<details>

<summary> Metrics Output for Models </summary>

```console
⭐️ Performance for: loopy-fudge
          R Squared Error: 0.5434174125466613
       Mean Squared Error: 523.423567982268
      Mean Absolute Error: 18.432238817050656
  Root Mean Squared Error: 22.87845204515087

⭐️ Performance for: sticky-frito
          R Squared Error: 0.752148321336189
       Mean Squared Error: 284.13569317262835
      Mean Absolute Error: 12.07890900951537
  Root Mean Squared Error: 16.856325019784958

⭐️ Performance for: swampy-cherry
          R Squared Error: -0.5802209706988015
       Mean Squared Error: 1811.555940617423
      Mean Absolute Error: 32.342102775897864
  Root Mean Squared Error: 42.562377055533716
```

</details>

After you are done, you can also save the (pickled) models for later (from your control plane or index 0 of the flux instance):

```bash
mkdir -p lammps-ml/results
cd lammps-ml/results
```
```python
from riverapi.main import Client

cli = Client('http://u2204-05:8080')

# Download model as pickle
for model_name in cli.models()['models']:
    # Saves to model-name>.pkl in pwd unless you provide a second arg, dest
    cli.download_model(model_name)

# Also save metrics and stats
import json
results = {}
for model_name in cli.models()['models']:
    results[model_name] = {
        "model": cli.get_model_json(model_name),
        "stats": cli.stats(model_name),
        "metrics": cli.metrics(model_name)
    } 

with open('post-train-models.json', 'w') as fd:
    fd.write(json.dumps(results, indent=3))
```

To save on fog -> quartz:

```bash
mkdir -p lammps-ml
cd lammps-ml
scp -r root@u2204-01:/home/flux/lammps-ml/results/* .

# then on quartz
mkdir -p lammps-ml
cd lammps-ml
scp -r sochat1@fog:/home/sochat1/lammps-ml/* .

# and then to your local machine
```
