#!/usr/bin/env python3

# After we've trained the models, run this script to run lammps again, and generate
# a testing set to generate predictions for. See how well we did.

import argparse
import os
import random
import shutil
import subprocess
import sys

from river import metrics
from riverapi.main import Client


def get_parser():
    parser = argparse.ArgumentParser(
        description="LAMMPS Testing (Serial)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="URL where ml-server is deployed",
        default="http://localhost",
    )
    parser.add_argument(
        "--workdir",
        default="/opt/lammps/examples/reaxff/HNS",
        help="Working directory to run lammps from.",
    )
    parser.add_argument(
        "--in",
        dest="inputs",
        default="in.reaxc.hns -nocite",
        help="Input and parameters for lammps",
    )
    parser.add_argument(
        "--nodes",
        help="number of nodes (N)",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--log",
        help="write log to path (keep in mind Singularity container is read only)",
        default="/tmp/lammps.log",
    )
    parser.add_argument(
        "--np",
        help="number of processes per node",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--min",
        help="min dimension for x,y,z",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--max",
        help="max dimension for x,y,z",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--iters",
        help="iterations to run of lammps",
        default=20,
        type=int,
    )
    return parser


def validate(args):
    if args.min < 1:
        sys.exit("Min value must be greater than or equal to 1")
    if args.min >= args.max:
        sys.exit("Max value must be greater than or equal to min")
    if args.max < 1:
        sys.exit("Max also needs to be positive >1. Also, we should never get here.")


def parse_time(line):
    line = line.rsplit(" ", 1)[-1]
    hours, minutes, seconds = line.split(":")
    return (int(hours) * 60 * 60) + (int(minutes) * 60) + int(seconds)


def main():
    parser = get_parser()

    # If an error occurs while parsing the arguments, the interpreter will exit with value 2
    args, _ = parser.parse_known_args()

    print(f"Preparing to run lammps and test models at {args.url}")

    # We need to be in this PWD with the experiment data
    # Yes, this assumes running in the container
    os.chdir(args.workdir)

    # Find the software we need
    mpirun = shutil.which("mpirun")
    lmp = shutil.which("lmp")

    if not mpirun or not lmp:
        sys.exit("Cannot find lmp or mpirun executable.")

    # Input files
    inputs = args.inputs.split(" ")

    # Connect to the server running here
    cli = Client(args.url)

    # Sanity check values
    validate(args)

    # Choose ranges to allow for each of x, y, and z. My computer is 32 cores so shouldn't go too big ;)
    # We are going to be lazy and just generate the ranges and choose randomly from them
    choices = list(range(args.min, args.max + 1))

    # https://riverml.xyz/latest/api/metrics/Accuracy/
    # Keep a listing actual and predictions (predictions namespaced by model)
    y_true = []
    y_pred = {}

    for i in range(args.iters):
        print(f"\n🧪️ Running iteration {i}")
        x = random.choice(choices)
        y = random.choice(choices)
        z = random.choice(choices)

        cmd = [
            mpirun,
            "-N",
            str(args.nodes),
            "--ppn",
            str(args.np),
            lmp,
            "-v",
            "x",
            str(x),
            "y",
            str(y),
            "z",
            str(z),
            "-log",
            args.log,
            "-in",
        ] + inputs
        print(" ".join(cmd))
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # This will hang until it's done
        # We could save the log here, but I'm just going to grab the time
        output, errors = p.communicate()
        line = [x for x in output.split("\n") if x][-1]
        if "total wall time" not in line.lower():
            print(f"Warning, there was an issue with iteration {i}")
            print(output)
            print(errors)
            continue
        seconds = parse_time(line)

        # Add to accuracy vector
        y_true.append(seconds)
        test_x = {"x": y, "y": y, "z": z}
        print(f"  Actual value is {seconds}")
        for model_name in cli.models()["models"]:
            pred = cli.predict(model_name, x=test_x)["prediction"]
            print(f"  Predicted value for {model_name} with {test_x} is {pred}")
            if model_name not in y_pred:
                y_pred[model_name] = []
            y_pred[model_name].append(pred)

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


if __name__ == "__main__":
    main()
