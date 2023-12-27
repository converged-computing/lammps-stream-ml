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

### TODO

These are next steps I want to do, after I have the automated builds.

- Update submission script to use flux instead.
- Maybe a script to consume API to get model as it is changing (and write somewhere for plotting)
