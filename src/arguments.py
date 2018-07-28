from argparse import ArgumentParser

BATCH_SIZE = 10
DATA_DIRECTORY = './dataset/bdd100k'
DATA_LIST_PATH = './dataset/list/train_list.txt'
INPUT_SIZE = '720,720'

def get_args():

    parser = ArgumentParser(description="Lane Instance Segmentation - BDD Dataset")

    # Training model hyperparameters
    # parser.add_argument(
    #     "--batch-size",
    #     "-b",
    #     type=int,
    #     default=100,
    #     help="Mini-batch size. Default: 100")
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default: 300")
    parser.add_argument(
        "--run-cuda",
        dest='cuda',
        default=True,
        help="CPU only.")

    parser.add_argument("--batch-size","-b",type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--gpu-select",'-g',type=str, default="0",
                        help="Select the GPU to run the program on")
    parser.add_argument("--run-name","-rn",type=str, default='run_def',
                        help="Choose the folder name to which the checkpoints should be saved")
    parser.add_argument("--mode","-m",type=str,default='train',
                        help="Choose to run train or test for the model")
    parser.add_argument('--load','-l',type=str,default=None,
                        help="Choose the checkpoint file name to test the model")


    return parser.parse_args()
