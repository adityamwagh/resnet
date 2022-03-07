# Deep Residual Networks
Deep Residual Networks for image classification on the CIFAR10 dataset.

Run this command to check whether everything is working as expected.

```
python main.py -w 1 -o sgd -d gpu -e 5 -lr 0.1 -dp data -wp weights.pth
```
See what these commands do.

```
usage: main.py [-h] -w WORKERS -o OPTIMISER -d DEVICE -e EPOCHS [-lr LEARNING_RATE] [-m MOMENTUM] [-wd WEIGHT_DECAY] -dp DATA_PATH -wp WEIGHT_PATH

optional arguments:
  -h, --help            show this help message and exit
  -w WORKERS, --workers WORKERS
                        number of workers for the data loader
  -o OPTIMISER, --optimiser OPTIMISER
                        optimizer for training
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
  -wp WEIGHT_PATH, --weight-path WEIGHT_PATH
                        path to weights of the trained model
```