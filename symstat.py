
import os
import logging
from logging.handlers import RotatingFileHandler

import torch
import numpy as np

from arg_utils import parse_args
from trainer import Trainer

def supress_log(lib_name):
    logging.getLogger(lib_name).setLevel(logging.INFO)


if __name__ == '__main__':

    args = parse_args()

    if args.module is None:
        raise ValueError("Provide a module to run! Use the (--help) flag")

    handlers = [
        logging.handlers.RotatingFileHandler(
            "logs/{}.log".format(args.module), maxBytes=1048576*5, backupCount=7),
        logging.StreamHandler()
    ]

    log_format = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            handlers=handlers, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO,
                            handlers=handlers, format=log_format)

    for lib_name in {"PIL", "matplotlib"}:
        supress_log(lib_name)

    log = logging.getLogger("root")

    log.info("Arguments: {}".format(args))

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if hasattr(args, "device") and args.device != "cpu":
        torch.cuda.manual_seed_all(args.seed)

    if args.module == "train":
        trainer = Trainer(args)
        if args.test:
            trainer.load()
            trainer.test()
        else:
            trainer.train(args.epochs)
    else:
        raise ValueError()