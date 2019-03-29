import argparse

def parse_args():
    """Parse args
    """
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("--verbose", dest="verbose", default=True,
                        action="store_true", help="set to true for verbose output")
    parser.add_argument("--seed", dest="seed", required=False,
                        default=42, help="random seed")

    subparsers = parser.add_subparsers(dest="module", help="module to run")

    train_parser = subparsers.add_parser("train", help="train a model")

    add_train_args(train_parser)

    return parser.parse_args()


def add_train_args(parser):
    parser.add_argument("--model", type=str,
                        choices=["sl"],
                        required=True)
    parser.add_argument("--model-id", dest="model_id",
                        type=str, required=True, help="id of the experiment")
    parser.add_argument("--num-workers", type=int, dest="num_workers", default=1,
                        help="number of workers to use")
    parser.add_argument("--batch-size", dest="batch_size",
                        type=int, default=2, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=1,
                        help="batch size for training")
    parser.add_argument("--device", type=str, default="cpu",
                        help="device to train model on")
    parser.add_argument("--test", action="store_true",
                        help="runs the model on the test set")
    parser.add_argument("--plot-aggr", dest="plot_aggr", action="store_true",
                        help="plots the aggregated posterior + prior")

    # Learning rates
    parser.add_argument("--learning-rate", type=float, dest="learning_rate",
                        help="learning rate for the optimizer", default=1e-3)

