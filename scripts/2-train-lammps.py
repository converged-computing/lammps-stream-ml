#!/usr/bin/env python3

# After the models are created with 1-create-models.py we can run
# this demo that will run lammps some number of times (in serial since I'm
# on my local machine) and use the matrix data for training the models it
# discovers.

from riverapi.main import Client
import argparse
import shutil
import random
import subprocess
import os
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="LAMMPS Training (Serial)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "url", nargs="?", help="URL where ml-server is deployed", default="localhost"
    )
    parser.add_argument(
        "--workdir",
        default="/opt/lammps/examples/reaxff/HNS",
        help="Working directory to run lammps from.",
    )
    parser.add_argument(
        "--in",
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

    print(f"Preparing to run lammps and train models at  {args.url}")

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

    for i in range(args.iters):
        print(f"\n🎄️ Running iteration {i}")
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

        # Send this to the server to train each model
        train_x = {"x": y, "y": y, "z": z}
        train_y = seconds
        for model_name in cli.models()["models"]:
            print(f"  Training {model_name} with {train_x} to predict {train_y}")
            cli.learn(model_name, x=train_x, y=train_y)


if __name__ == "__main__":
    main()
