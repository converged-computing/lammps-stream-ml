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

And we can incrementally use the same API to get the model itself, and show the plot / change over time. We can try these out with different models, for fun!

## 1. Machine Learning Server

Let's first build the Dockerfile.

```bash
docker build -t lammps-ml .
```

And run to see the server starting:

```bash
docker run -it lammps-ml
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
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

This is going to be cool! I literally just got this idea falling asleep and jumped up to prototype this quick thing - will do the following items tomorrow!
I am not going to pull another stay up until the wee hours thing... XD
This is so simple it just might work!

### TODO

- Test running local container with model to predict Y (time) from some x, y, z
- When that works, update settings params to come from environment
- Put into a Kubernetes object, with environment vars from "secrets"
- Try doing the same, exposing via ingress service
- Write small dummy lammps submission script (likely in another container for now) that submits data to API
- Write script to consume API to get model results (and write somewhere for plotting)
