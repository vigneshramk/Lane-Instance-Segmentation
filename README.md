# Drivable Area Segmentation and Parameterization

PyTorch implementation of our framework for Drivable Area Segmentation and Parameterization using the [*ENet Architecture*](https://arxiv.org/abs/1606.02147) applied to the [*Berkeley Deep Drive Dataset](http://bdd-data.berkeley.edu) containing 100K drivable area maps.

## Usage

Run [``main.py``], the script used to run the training/testing of the ENet model

```
python main.py [-h] [--num-epochs NUM_EPOCHS] [--learning-rate LEARNING_RATE]
               [--lr-decay LR_DECAY] [--lr-decay-epochs LR_DECAY_EPOCHS]
               [--weight-decay WEIGHT_DECAY] [--epochs EPOCHS]
               [--run-cuda CUDA] [--batch-size BATCH_SIZE]
               [--data-dir DATA_DIR] [--data-list DATA_LIST]
               [--input-size INPUT_SIZE] [--gpu-select GPU_SELECT]
               [--run-name RUN_NAME] [--mode MODE] [--load LOAD]

```

For help on the optional arguments run: ``python main.py -h`` 

Take a look at [``src/arguments.py``] to check for the default arguments

### Training Arguments (Example):

```
python main.py --gpu-select 0 --batch-size 10 --run-name train_example

```
(OR) Equivalently,

```
python main.py -g 0 -b 10 -rn train_example

```

### Testing Arguments (Example):

Note: This example uses a model checkpoint stored in [``saved_models/run4/``]. If the model is elsewhere, provide the path to the model instead.

```
python main.py --gpu-select 0 --batch-size 5 --run-name test_example --mode test --load saved_models/run4/checkpoint_20.h5

```
(OR) Equivalently,

```
python main.py -g 0 -b 5 -rn test_example -m test -l saved_models/run4/checkpoint_20.h5

```

## Code Organization

### Directories

[``dataset``] contains the dataloader codes, the dataset (not on the repository, download it yourself with the instructions given below) and list of labels that are used by the dataloader
[``src``] contains the training and testing codes, the metrics that are used for evaluation in [``src/metrics``], the arguments and the helper functions used by the other codes.
[``models``] contains the ENet model architecture code definitions
[``saved_models``] contains the saved model checkpoints of our implementation inside a folder corresponding to the run_name provided at run time.