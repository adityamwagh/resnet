
# Deep Residual Networks for Image Classification

Pytorch implementation of ResNet18 for CIFAR10 classification.

## Authors

It's a joint work between Aditya, Vijay and Yash

- [Aditya Wagh](https://www.github.com/adityamwagh)
- [Vijayraj Gohil](https://www.github.com/vraj130)
- [Yash Patel](https://www.github.com/yyashpatel)


## Setup

Install either of [Anaconda](https://www.anaconda.com/), [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge/releases/tag/4.11.0-4).
All of them provide the `conda` package manager for python which we will use to install our project packages.

The code has beed tested using the following python packages.

- `torch >= v1.10`
- `torchvision >= v0.11` 

Create a virtual environment to install project dependencies.

```bash
conda create -n torch
```

Activate the virtual environment.
```
conda activate torch
```

Install **PyTorch** and **Torchvision** using from the pytorch channel.
```
conda install pytorch torchvision matplotlib cudatoolkit=11.3 -c pytorch
```
    
## Usage

| Argument                              | Type       | Default value  | Description                                           |
| :-------------------------------------| :----------| :--------------| :-----------------------------------------------------|
| `-en` or `--experiment-number`        | `required` | `1`            | **Required**. Your API key                            |
| `-e`  or `--epochs`                   | `required` | `120`          | **Required**. Your API key                            |
| `-o`  or `--optimizer`                | `required` | `N/A`          | **Required**. Your API key                            |
| `-d`  or `--device`                   | `required` | `gpu`          | **Required**. Your API key                            |
| `-lr` or `--learning-rate`            | `required` | `0.1`          | **Required**. Your API key                            |
| `-mo`  or `--momentum`                | `required` | `0.9`          | **Required**. Your API key                            |
| `-wd` or `--weight-decay`             | `required` | `5e-4`         | **Required**. Your API key                            |
| `-dp` or `--data-path`                | `required` | `./data`       | **Required**. Your API key                            |
| `-b`  or `--blocks`                   | `required` | `required`     | **Required**. Your API key                            |
| `-c`  or `--channels`                 | `required` | `required`     | **Required**. Your API key                            |
| '-m'  or '--model'                    | `required` | 'resnet'       | **Required**. Your API key                            |

```bash
python main.py \
    --experiment-number 1 \ 
    --optimizer sgd \ 
    --data-path ./data \
    --blocks 2 2 2 2 \
    --channel 54 96 188 324

```
See what these commands do.

```
usage: main.py [-h] -en EXPERIMENT_NUMBER -o OPTIMISER [-d DEVICE] [-e EPOCHS] [-lr LEARNING_RATE] [-m MOMENTUM] [-wd WEIGHT_DECAY] -dp DATA_PATH -b BLOCKS BLOCKS BLOCKS BLOCKS -c CHANNELS CHANNELS CHANNELS CHANNELS

  optional arguments:
  -h, --help            show this help message and exit
  -en EXPERIMENT_NUMBER, --experiment_number EXPERIMENT_NUMBER
                          number to track the different experiments
  -o OPTIMISER, --optimiser OPTIMISER
                          optimizer for training
  -m MODEL, --model MODEL
                          model to train
  -d DEVICE, --device DEVICE
                          device to train on
  -e EPOCHS, --epochs EPOCHS
                          number of epochs to train for
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                          learning rate for the optimizer
  -m MOMENTUM, --momentum MOMENTUM
                          momentum value for optimizer if applicable
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                          weight decay value for the optimizer if applicable
  -dp DATA_PATH, --data-path DATA_PATH
                          path to the dataset
  -b BLOCKS BLOCKS BLOCKS BLOCKS, --blocks BLOCKS BLOCKS BLOCKS BLOCKS
                          number of blocks in each layer
  -c CHANNELS CHANNELS CHANNELS CHANNELS, --channels CHANNELS CHANNELS CHANNELS CHANNELS
                          number of channels in each layer 

```
