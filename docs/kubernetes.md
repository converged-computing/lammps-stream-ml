# Kubernetes

Here I want to test deploying the same containers to Kubernetes. This means that we will expose the API as a service, and provide credentials via secrets (not necessary, but will be good to test not using the default ones). This assumes running from the [root of the repository](../)

## Cluster

First, create a test cluster.

```bash
kind create cluster --config ./k8s/kind-config.yaml
```

Note that I struggled with creating ingress for a while because Kind needs additions for it to work (in that file).

### Ingress

We then want to create a service to access the TBA ml-service.

```bash
kubectl apply -f k8s/ingress.yaml
```
```console
$ kubectl describe ingress
Name:             ml-ingress
Labels:           <none>
Namespace:        default
Address:          
Ingress Class:    <none>
Default backend:  <default>
Rules:
  Host        Path  Backends
  ----        ----  --------
  localhost   
              /   ml-service:8080 (<none>)
Annotations:  <none>
Events:       <none>
```

We are then going to apply [ingress-nginx](https://kind.sigs.k8s.io/docs/user/ingress/#ingress-nginx).

```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=90s
```

### Deployment

Deploy the machine learning server.

```bash
kubectl apply -f k8s/server-deployment.yaml
```

Note that we have hard coded secrets, which is OK for local testing, but you should update these to a secret proper for anything more than that.
If the ingress and deployment are successful, you should be able to do the following to localhost:

```bash
$ curl -ks localhost/api/ | jq
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

Since we want to run lammps, let's test writing a container and script to do that next. Our current container has the client for river, but not lammps.
Let's combine the two.

### Create Models

Let's create three empty models. Since we know the service is running, we can now
pull the same lammps container that we will use for the jobs (and run a script to create models, which is prepared inside).
We will need this container anyway to run lammps, might as well use it for other things!

```bash
singularity pull docker://ghcr.io/converged-computing/lammps-stream-ml:lammps
```

Here is how to create the models for the running server. The names will be funny but largely don't matter - we can get them programmatically later.

```bash
# Assumes service running on localhost directory (first parameter, default)
singularity exec lammps-stream-ml_lammps.sif python3 /code/1-create-models.py
```
```console
Preparing to create models for client URL http://localhost
Created model expressive-cupcake
Created model confused-underoos
Created model doopy-platanos
```

### Train LAMMPS

Now let's run our script that is going to (in a loop) run lammps and send the results to the server to train.
This is still a local example, so we aren't doing this via submitting jobs to flux (but easily could, and will).
Note that this script has a lot of options you can customize, so take a look first.

```bash
singularity exec lammps-stream-ml_lammps.sif python3 /code/2-train-lammps.py --help
```

To use the defaults (range of 1 to 8 for x,y,z and 20 iterations with 1 node and 4 processes per node):

```bash
singularity exec lammps-stream-ml_lammps.sif python3 /code/2-train-lammps.py
```
```console
...
🎄️ Running iteration 2
/usr/bin/mpirun -N 1 --ppn 4 /usr/bin/lmp -v x 3 y 3 z 7 -log /tmp/lammps.log -in in.reaxc.hns -nocite
  Training confused-underoos with {'x': 3, 'y': 3, 'z': 7} to predict 20
  Training doopy-platanos with {'x': 3, 'y': 3, 'z': 7} to predict 20
  Training expressive-cupcake with {'x': 3, 'y': 3, 'z': 7} to predict 20

🎄️ Running iteration 3
/usr/bin/mpirun -N 1 --ppn 4 /usr/bin/lmp -v x 8 y 6 z 4 -log /tmp/lammps.log -in in.reaxc.hns -nocite
  Training confused-underoos with {'x': 6, 'y': 6, 'z': 4} to predict 51
  Training doopy-platanos with {'x': 6, 'y': 6, 'z': 4} to predict 51
  Training expressive-cupcake with {'x': 6, 'y': 6, 'z': 4} to predict 51

🎄️ Running iteration 4
/usr/bin/mpirun -N 1 --ppn 4 /usr/bin/lmp -v x 4 y 4 z 4 -log /tmp/lammps.log -in in.reaxc.hns -nocite
  Training confused-underoos with {'x': 4, 'y': 4, 'z': 4} to predict 26
  Training doopy-platanos with {'x': 4, 'y': 4, 'z': 4} to predict 26
  Training expressive-cupcake with {'x': 4, 'y': 4, 'z': 4} to predict 26
```

This will run your lammps to generate training data, and send it to the server, training each (of three) models.


### Predict LAMMPS

Now let's generate more data, but this time, compare the actual time with each model prediction. This script is very similar but calls a different API function.

```bash
singularity exec lammps-stream-ml_lammps.sif python3 /code/3-predict-lammps.py
```

And then you'll run lammps for some number of iterations (defaults to 20) and calculate an accuracy for each model.
Note that there are a lot of metrics you can see [here](https://riverml.xyz/latest/api/metrics/Accuracy/).
The server itself also stores basic metrics, but we are doing this manually so it's a hold out test set.

## Clean Up

When you are done, clean up.

```bash
kind delete cluster
```
