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

At this point we will have an ML server that can serve predictions for a model. We could incrementally use the same API to get the model itself, and show the plot / change over time. We could also do a separate set of runs (test data) and then use that to see how well we did. We can try these out with different models, for fun! Since this project will have several stages, I'll organize them into the [docs](docs) directory.

 - [Building Containers](docs/containers.md): the first prototype to see if the idea had feet, running a server and having a client create models, train, and predict using it.
 - [Kubernetes](docs/kubernetes.md): deploying the ml-server to kubernetes and running actual lammps alongside it, in a Singularity container, training and testing and calculating accuracy for three models.
 - [Usernetes](docs/usernetes.md): the same, but move into usernetes with flux.

## Results

The prototype results are in! The entire method worked really nicely on Usernetes + bare metal (e.g., Flux on VMs) and although I didn't take much time into the ML bit, we have some fun plots that demonstrate different kinds of regression to predict LAMMPS runtime from the x,y,z coordinates, trained on 1000 points for each of the following parameters:

- x: between 1 and 32
- y: between 1 and 8
- z: between 1 and 16

And then tested (on a new set run on the same cluster) of 250 points.

![results/lammps-ml/loopy-fudge-pa-regression.png](results/lammps-ml/loopy-fudge-pa-regression.png)
![results/lammps-ml/sticky-frito-linear-regression.png](results/lammps-ml/sticky-frito-linear-regression.png)
![results/lammps-ml/swampy-cherry-bayesian-linear-regression.png](results/lammps-ml/swampy-cherry-bayesian-linear-regression.png)

For someone that doesn't do ML and just casually through this together, I'm less concerned with the result, but pretty proud that I pulled the whole thing off! At a high level, this is a great demonstration of:

1. Running a simulation on bare metal HPC alongside a service
2. Sending results to the service as you go (in this case, ML training points)
3. Using the service to get updated info about the model on demand
4. Doing a second phase (e.g., hold out testing) with your trained model.

And that's it! I think we might have done better to predict log time, but I don't want to go back and do it again. But it would actually be interesting to have this setup in a live running queue alongside a set of ensemble workloads that can generate predictions about the runtime (and schedule accordingly).
