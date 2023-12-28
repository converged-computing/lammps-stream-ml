# Usernetes

We are going to try and reproduce the setup in [kubernetes](kubernetes.md) but with usernetes on a private cluster, and running lammps via submit jobs instead of otherwise. The details of the cluster are left out here, but you can assume you need to ssh into it and have dual access to usernetes and flux.

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
$ kubectl  get pods -o wide --watch
NAME                         READY   STATUS    RESTARTS   AGE   IP           NODE           NOMINATED NODE   READINESS GATES
ml-server-7bb7fc4764-b7rjq   1/1     Running   0          31s   10.244.3.2   u7s-u2204-04   <none>           <none>
```

And then we can ping the API at port 8080.

```bash
$ curl -sk u2204-04:8080/api/ | jq
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
ingress.yaml  lammps-stream-ml_lammps.sif  server-deployment.yaml
```

Here is how to create the models for the running server. The names will be funny but largely don't matter - we can get them programmatically later. We do need to tell the script the hostname the server is running at:

```bash
singularity exec lammps-stream-ml_lammps.sif python3 /code/1-create-models.py http://u2204-04:8080/
```
```console
Preparing to create models for client URL http://u2204-04:8080/
Created model rainbow-kerfuffle
Created model swampy-milkshake
Created model misunderstood-punk
```

### Train LAMMPS

Now let's run our script that is going to (via flux submit) run lammps and send the results to the server to train. This requires a different setup than our initial testing because we need the script to submit the flux jobs and target the container, meaning that we need to make two calls within this script to the singularity container - to run lammps and then submit the result. Note that this also means the script will be run directly on the host (and does not require any ML libraries). Let's download it first:

```bash
# Only requires standard library
wget https://raw.githubusercontent.com/converged-computing/lammps-stream-ml/main/scripts/2-run-lammps-flux.py
```
```console
python3 2-train-lammps-flux.py --help
```

Also if you need to clear the cache:

```bash
flux exec singularity cache clean --force
```

Now let's train, allowing each parameter to range between these values (the script controls this)

 - x: 1 to 32
 - y: 1 to 8
 - z: 1 to 16

This is a bit arbitrary based on my manual testing, but I had to cap the max runtime. with some strategy. The flux submit is also nice because I can submit a ton of jobs and then just leave them to run.
Let's do a test run first (one iteration, and flux run so we see everything).

```bash
container=/home/flux/lammps-ml/lammps-stream-ml_lammps.sif
python3 2-train-lammps-flux.py --container $container --flux-cmd run --np 48 --nodes 6 --workdir /opt/lammps/examples/reaxff/HNS --x-min 1 --x-max 32 --y-min 1 --y-max 8 --z-min 1 --z-max 16 --iters 1 http://u2204-04:8080/
```
```console
Preparing to run lammps and train models with /home/flux/lammps-ml/lammps-stream-ml_lammps.sif

🎄️ Running iteration 0 with chosen x: 4 y: 4 z: 1
         flux => /usr/bin/flux run -N 6 --ntasks 48 -c 1 -o cpu-affinity=per-task
  singularity => /usr/bin/singularity exec --pwd /opt/lammps/examples/reaxff/HNS /home/flux/lammps-ml/lammps-stream-ml_lammps.sif /usr/bin/lmp -v x 4 -v y 14 -v z 6 -log /tmp/lammps.log -in in.reaxc.hns -nocite
       result => Lammps run took 16 seconds
      command => /usr/bin/singularity exec /home/flux/lammps-ml/lammps-stream-ml_lammps.sif python3 /code/2-send-train-result.py --x 4 --y 14 --z 6 --time 16 http://u2204-04:8080/
Preparing to send LAMMPS data to http://u2204-04:8080/
  Training misunderstood-punk with {'x': 4, 'y': 4, 'z': 1} to predict 16
  Training rainbow-kerfuffle with {'x': 4, 'y': 4, 'z': 1} to predict 16
  Training swampy-milkshake with {'x': 4, 'y': 4, 'z': 1} to predict 16
```

That was a small z dimension, so very fast! Now let's run in a loop. Since each run takes all the resources, it's OK to do these in serial with flux run (instead of flux submit). Let's start with 20 iterations.

```bash
container=/home/flux/lammps-ml/lammps-stream-ml_lammps.sif
python3 2-train-lammps-flux.py --container $container --flux-cmd run --np 48 --nodes 6 --workdir /opt/lammps/examples/reaxff/HNS --x-min 1 --x-max 32 --y-min 1 --y-max 32 --z-min 1 --z-max 32 --iters 20 http://u2204-04:8080/
```

You can watch the output to see chosen parameters and output (as shown in the example). This will run your lammps to generate training data, and send it to the server, training each (of three) models. 
Note that from a different console you can monitor progress. At this point, we won't see any predictions, but we will see changing stats and the learn endpoint being called!

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

After running this a bunch of times (how many I'm not sure, maybe 100?) We will use predict on it to see how well our models do. I'm currently doing a final test of problem sizes to decide on a final min/max for each, will be running the above (for our more formal results) soon.

### Predict LAMMPS

**TODO**

