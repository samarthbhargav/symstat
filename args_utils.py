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

    