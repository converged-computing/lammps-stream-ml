#!/usr/bin/env python3

# After the models are created with 1-create-models.py we can run
# this demo that will run lammps some number of times (in serial since I'm
# on my local machine) and use the matrix data for training the models it
# discovers.

# This script is run a bit differently - it is expected to be run on the host, and then
# interacts with the container to run lammps, get the result, and then submit the result
# to the machine learning server (using the same container). So we have taken 2-train-lammps.py
# and split into two scripts (still in one container) to account for being on the host.

import argparse
import sys

from riverapi.main import Client


def get_parser():
    parser = argparse.ArgumentParser(
        description="Send LAMMPS Training Result",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="URL where ml-server is deployed",
        default="http://localhost",
    )
    parser.add_argument(
        "--time",
        help="time that job took to run",
        type=int,
    )

    # Mins and maxes for each parameter - I decided to allow up to 32, 16, 16 for testing.
    parser.add_argument(
        "--x",
        help="x dimension",
        type=int,
    )
    parser.add_argument(
        "--y",
        help="y dimension",
        type=int,
    )
    parser.add_argument(
        "--z",
        help="z dimension",
        type=int,
    )

    return parser


def validate(args):
    if not args.x or not args.y or not args.z or not args.time:
        sys.exit("Each of --x, --y, and --z, and --time are required")


def main():
    parser = get_parser()

    # If an error occurs while parsing the arguments, the interpreter will exit with value 2
    args, _ = parser.parse_known_args()

    print(f"Preparing to send LAMMPS data to {args.url}")

    # Connect to the server running here
    cli = Client(args.url)

    # Send this to the server to train each model
    train_x = {"x": args.x, "y": args.y, "z": args.z}
    train_y = args.time
    for model_name in cli.models()["models"]:
        print(f"  Training {model_name} with {train_x} to predict {train_y}")
        cli.learn(model_name, x=train_x, y=train_y)


if __name__ == "__main__":
    main()
