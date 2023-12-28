#!/usr/bin/env python3

# Make a prediction after we have a result.
# or get results / metrics

import argparse
import sys

from riverapi.main import Client


def get_parser():
    parser = argparse.ArgumentParser(
        description="Make prediction or get summary metrics",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="URL where ml-server is deployed",
        default="http://localhost",
    )
    subparsers = parser.add_subparsers(
        help="actions",
        title="actions",
        description="actions",
        dest="command",
    )

    predict = subparsers.add_parser("predict", description="make prediction")
    metrics = subparsers.add_parser("metrics", description="get metrics for all models")
    predict.add_argument(
        "--x",
        help="x dimension",
        type=int,
    )
    predict.add_argument(
        "--y",
        help="y dimension",
        type=int,
    )
    predict.add_argument(
        "--z",
        help="z dimension",
        type=int,
    )
    return parser


def validate(args):
    if not args.x or not args.y or not args.z:
        sys.exit("Each of --x, --y, and --z are required")


def main():
    parser = get_parser()

    # If an error occurs while parsing the arguments, the interpreter will exit with value 2
    args, _ = parser.parse_known_args()

    # If we are making a prediction, all params required
    if args.command == "predict":
        validate(args)

    # Connect to the server running here
    cli = Client(args.url)

    if args.command == "predict":
        make_prediction(cli, args)


def make_prediction(cli, args):
    """
    Make a prediction.
    """
    test_x = {"x": args.x, "y": args.y, "z": args.z}
    for model_name in cli.models()["models"]:
        pred = cli.predict(model_name, x=test_x)["prediction"]
        print(f"Model {model_name} predicts {pred}")


def get_metrics(cli):
    """
    Print metrics for all models
    """
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
