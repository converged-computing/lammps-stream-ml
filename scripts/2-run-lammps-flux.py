#!/usr/bin/env python3

# After the models are created with 1-create-models.py we can run
# this demo that will run lammps some number of times (in serial since I'm
# on my local machine) and use the matrix data for training the models it
# discovers. This script has combined the test and train functions to be
# able to use shared logic to run lammps (to avoid duplication)

# This script requires the riverapi
# pip3 install riverapi

import argparse
import random
import shutil
import subprocess
import json
import sys

from riverapi.main import Client
from river import metrics

# Find the software we need, flux and singularity
flux = shutil.which("flux")
singularity = shutil.which("singularity")

if not flux or not singularity:
    sys.exit("Cannot find flux or singularity executable.")


def get_parser():
    parser = argparse.ArgumentParser(
        description="LAMMPS Run (Train or Test) Flux",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        help="actions",
        title="actions",
        description="actions",
        dest="command",
    )

    # print version and exit
    train = subparsers.add_parser("train", description="train models")
    test = subparsers.add_parser(
        "predict",
        description="test models by making predictions and comparing to truth",
    )

    for command in [train, test]:
        command.add_argument(
            "--url",
            help="URL where ml-server is deployed",
            default="http://localhost",
        )
        command.add_argument(
            "--workdir",
            default="/opt/lammps/examples/reaxff/HNS",
            help="Working directory to run lammps from.",
        )
        command.add_argument(
            "--container",
            help="Path to container to run with lammps",
        )
        command.add_argument(
            "--in",
            dest="inputs",
            default="in.reaxc.hns -nocite",
            help="Input and parameters for lammps",
        )
        command.add_argument(
            "--nodes",
            help="number of nodes (N)",
            default=1,
            type=int,
        )
        command.add_argument(
            "--log",
            help="write log to path (keep in mind Singularity container is read only)",
            default="/tmp/lammps.log",
        )
        command.add_argument(
            "--np",
            help="number of processes per node",
            default=4,
            type=int,
        )

        # Mins and maxes for each parameter - I decided to allow up to 32, 32, 32 for testing.
        # On the cluster with cpu affinity set this is 4 minutes 41 seconds
        command.add_argument(
            "--x-min",
            dest="x_min",
            help="min dimension for x",
            default=1,
            type=int,
        )
        command.add_argument(
            "--x-max",
            dest="x_max",
            help="max dimension for x",
            default=32,
            type=int,
        )
        command.add_argument(
            "--y-min",
            dest="y_min",
            help="min dimension for y",
            default=1,
            type=int,
        )
        command.add_argument(
            "--y-max",
            dest="y_max",
            help="max dimension for y",
            default=32,
            type=int,
        )
        command.add_argument(
            "--z-min",
            dest="z_min",
            help="min dimension for z",
            default=1,
            type=int,
        )
        command.add_argument(
            "--z-max",
            dest="z_max",
            help="max dimension for z",
            default=32,
            type=int,
        )
        command.add_argument(
            "--iters",
            help="iterations to run of lammps",
            default=20,
            type=int,
        )
    return parser


def validate(args):
    for dim, min_value, max_value in [
        ["x", args.x_min, args.x_max],
        ["y", args.y_min, args.y_max],
        ["z", args.z_min, args.z_max],
    ]:
        if min_value < 1:
            sys.exit(f"Min value for {dim} must be greater than or equal to 1")
        if min_value >= max_value:
            sys.exit(f"Max value for {dim} must be greater than or equal to min")
        if max_value < 1:
            sys.exit(
                f"Max for {dim} also needs to be positive >1. Also, we should never get here."
            )


def parse_time(line):
    line = line.rsplit(" ", 1)[-1]
    hours, minutes, seconds = line.split(":")
    return (int(hours) * 60 * 60) + (int(minutes) * 60) + int(seconds)


def run_lammps(args):
    """
    Shared function to run lammps for train or testing.

    We return (yield) chosen x,y,z and time in seconds as we run
    """
    # Input files
    inputs = args.inputs.split(" ")

    # Sanity check values
    validate(args)

    # Choose ranges to allow for each of x, y, and z.
    x_choices = list(range(args.x_min, args.x_max + 1))
    y_choices = list(range(args.y_min, args.y_max + 1))
    z_choices = list(range(args.z_min, args.z_max + 1))

    for i in range(args.iters):
        x = random.choice(x_choices)
        y = random.choice(y_choices)
        z = random.choice(z_choices)
        print(f"\n🎄️ Running iteration {i} with chosen x: {x} y: {y} z: {z}")

        # flux run -N 6 --ntasks 48 -c 1 -o cpu-affinity=per-task singularity exec --pwd /opt/lammps/examples/reaxff/HNS $container /usr/bin/lmp -v x 32 -v y 8 -v z 16 -in in.reaxc.hns
        # Separate into flux command and singularity command for printing
        flux_cmd = [
            flux,
            "run",
            "-N",
            str(args.nodes),
            "--ntasks",
            str(args.np),
            # These aren't exposed as options because we pretty much always want them
            "-c",
            "1",
            "-o",
            "cpu-affinity=per-task",
        ]
        singularity_cmd = [
            singularity,
            "exec",
            "--pwd",
            args.workdir,
            args.container,
            # This is where lammps is installed in the container, this should not change
            "/usr/bin/lmp",
            "-v",
            "x",
            str(x),
            "-v",
            "y",
            str(y),
            "-v",
            "z",
            str(z),
            "-log",
            args.log,
            "-in",
        ] + inputs

        print("         flux => " + " ".join(flux_cmd))
        print("  singularity => " + " ".join(singularity_cmd))
        cmd = flux_cmd + singularity_cmd
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # This will hang until it's done, either submit or run
        # We could save the log here, but I'm just going to grab the time
        output, errors = p.communicate()
        line = [x for x in output.split("\n") if x][-1]

        # Note this is currently written to run experiments, meaning we use all resources available
        # for each run, and can just wait for the run and parse output. If you want to use flux submit,
        # you can instead write each to a log file, read the log file, and parse the same.
        if "total wall time" not in line.lower():
            print(f"Warning, there was an issue with iteration {i}")
            print(output)
            print(errors)
            continue

        seconds = parse_time(line)
        print(f"       result => Lammps run took {seconds} seconds")
        yield x, y, z, seconds


def make_prediction(cli, args, x, y, z):
    """
    Make a prediction.
    """
    test_x = {"x": x, "y": y, "z": z}
    for model_name in cli.models()["models"]:
        pred = cli.predict(model_name, x=test_x)["prediction"]
        print(f"Model {model_name} predicts {pred}")
        yield model_name, pred


def submit_train_result(cli, args, x, y, z, seconds):
    """
    Submit a training result
    """
    print(f"Preparing to send LAMMPS data to {args.url}")

    # Send this to the server to train each model
    train_x = {"x": x, "y": y, "z": z}
    for model_name in cli.models()["models"]:
        print(f"  Training {model_name} with {train_x} to predict {seconds}")
        res = cli.learn(model_name, x=train_x, y=seconds)
        if "successful learn" not in res.lower():
            print(f"Issue with learn: {res}")


def show_metrics(cli, y_true, y_pred):
    # When we are done, calculate metrics for each
    for model_name in cli.models()["models"]:
        # Mean squared error
        mse_metric = metrics.MSE()

        # Root mean squared error
        rmse_metric = metrics.RMSE()

        # Mean absolute error
        mae_metric = metrics.MAE()

        # Coefficient of determination () score - r squared
        # proportion of the variance in the dependent variable that is predictable from the independent variable(s)
        r2_metric = metrics.R2()

        for yt, yp in zip(y_true, y_pred[model_name]):
            mse_metric.update(yt, yp)
            rmse_metric.update(yt, yp)
            mae_metric.update(yt, yp)
            r2_metric.update(yt, yp)

        print(f"\n⭐️ Performance for: {model_name}")
        print(f"          R Squared Error: {r2_metric.get()}")
        print(f"       Mean Squared Error: {mse_metric.get()}")
        print(f"      Mean Absolute Error: {mae_metric.get()}")
        print(f"  Root Mean Squared Error: {rmse_metric.get()}")

        # print("  => Model:")
        # print(cli.get_model_json(model_name))
        # print("  => Metrics:")
        # print(cli.metrics(model_name))
        # print("  => Stats:")
        # print(cli.stats(model_name))


def main():
    parser = get_parser()

    # If an error occurs while parsing the arguments, the interpreter will exit with value 2
    args, _ = parser.parse_known_args()

    # I actually don't think I need this check.
    if args.command not in ["train", "predict"]:
        sys.exit(f"{args.command} is not recognized.")

    print(f"Preparing to run lammps and {args.command} models with {args.container}")

    # Connect to the server running here
    cli = Client(args.url)

    # Do a test to the client
    res = cli.info()
    print(json.dumps(res, indent=4))

    # If we are predicting, we will save true / predicted values
    # https://riverml.xyz/latest/api/metrics/Accuracy/
    # Keep a listing actual and predictions (predictions namespaced by model)
    y_true = []
    y_pred = {}

    for x, y, z, seconds in run_lammps(args):
        if args.command == "train":
            submit_train_result(cli, args, x, y, z, seconds)
        else:
            # Add to accuracy vector
            y_true.append(seconds)

            # Make a prediction
            for model_name, pred in make_prediction(cli, args, x, y, z):
                if model_name not in y_pred:
                    y_pred[model_name] = []
                y_pred[model_name].append(pred)

    # When we are finished running, if we are predicting, give final results
    if args.command == "predict":
        show_metrics(cli, y_true, y_pred)


if __name__ == "__main__":
    main()
