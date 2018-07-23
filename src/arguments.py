from argparse import ArgumentParser


def get_args():

    parser = ArgumentParser()

    # Training model hyperparameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="Mini-batch size. Default: 100")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default: 100")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=5e-4,
        help="The learning rate. Default: 5e-4")
    parser.add_argument(
        "--lr-decay",
        "-lrd",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.1")
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=50,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 50")
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4")

    return parser.parse_args()