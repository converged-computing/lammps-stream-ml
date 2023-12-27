#!/usr/bin/env python3

# After the models are created with 1-create-models.py we can run
# this demo that will run lammps some number of times (in serial since I'm
# on my local machine) and use the matrix data for training the models it
# discovers.

# This script is run a bit differently - it is expected to be run on the host, and then
# interacts with the container to run lammps, get the result, and then submit the result
# to the machine learning server (using the same container). So we have taken 2-train-lammps.py
# and split into two scripts (still in one container) to account for being on the host.

# I had to split them apart because:
# 1. This script needs to CALL the container, and have access to flux on the host
# 2. This script (on the host) cannot expect to have river installed

# Usage to run on cluster (step 1)
# python3 /home/flux/lammps-ml/2-train-lammps-flux.py --container $container


# flux run -N 6 --ntasks 48 -c 1 -o cpu-affinity=per-task singularity exec --pwd /opt/lammps/examples/reaxff/HNS $container python3 /code/2-train-lammps.py --x-min 1 --x-max 32 --y-min 1 --y-max 8 --z-min 1 --z-max 8 --iters 1 --np 48 --nodes 6 http://u2204-04:8080/


import argparse
import random
import shutil
import subprocess
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="LAMMPS Training (Flux)",
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
        "--container",
        help="Path to container to run with lammps",
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
        "--flux-cmd",
        dest="flux_cmd",
        help="The flux command to use (e.g., run or submit)",
        default="run",
    )
    parser.add_argument(
        "--np",
        help="number of processes per node",
        default=4,
        type=int,
    )

    # Mins and maxes for each parameter - I decided to allow up to 32, 16, 16 for testing.
    parser.add_argument(
        "--x-min",
        dest="x_min",
        help="min dimension for x",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--x-max",
        dest="x_max",
        help="max dimension for x",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--y-min",
        dest="y_min",
        help="min dimension for y",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--y-max",
        dest="y_max",
        help="max dimension for y",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--z-min",
        dest="z_min",
        help="min dimension for z",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--z-max",
        dest="z_max",
        help="max dimension for z",
        default=16,
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


def main():
    parser = get_parser()

    # If an error occurs while parsing the arguments, the interpreter will exit with value 2
    args, _ = parser.parse_known_args()

    print(f"Preparing to run lammps and train models with {args.container}")

    # Find the software we need, flux and singularity
    flux = shutil.which("flux")
    singularity = shutil.which("singularity")

    if not flux or not singularity:
        sys.exit("Cannot find flux or singularity executable.")

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
        cmd = [
            flux,
            args.flux_cmd,
            "-N",
            str(args.nodes),
            "--ntasks",
            str(args.np),
            # These aren't exposed as options because we pretty much always want them
            "-c",
            "1",
            "-o",
            "cpu-affinity=per-task",
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

        # This will hang until it's done, either submit or run
        # We could save the log here, but I'm just going to grab the time
        output, errors = p.communicate()
        line = [x for x in output.split("\n") if x][-1]

        # Note this is currently written to run experiments, meaning we use all resources available
        # for each run, and can just wait for the run and parse output. If you want to use flux submit,
        # you can instead write each to a log file, read the log file, and parse the same.
        if args.flux_cmd == "run":
            if "total wall time" not in line.lower():
                print(f"Warning, there was an issue with iteration {i}")
                print(output)
                print(errors)
                continue

            seconds = parse_time(line)
            print(f"Lammps run took {seconds} seconds")
            print("TODO add command to submit to server here")
            import IPython

            IPython.embed()
            cmd = [
                singularity,
                "exec",
                args.container,
                "python3",
                "/code/2-send-train-result.py",
                "--x",
                x,
                "--y",
                y,
                "--z",
                z,
                "--time",
                seconds,
                args.url,
            ]
            print(" ".join(cmd))
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output, errors = p.communicate()
            print(output)
            print(errors)
            continue

        print(output)
        print(errors)


if __name__ == "__main__":
    main()
