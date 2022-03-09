import argparse
import time
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from project1_model import project1_model

# Defining the Training Loop
def train_test_model(epochs, train_loader, test_loader, model, loss_fn, optimizer, train_loss_values, train_accuracy):

    model.train()
    for epoch in range(1, epochs + 1):
        with tqdm(train_loader, unit="batches") as train_data_loader:

            running_loss = 0
            correct = 0
            total = 0

            for imgs, labels in train_data_loader:

                train_data_loader.set_description(f"Training: Epoch {epoch}/{epochs}")

                X, y = imgs.to(device), labels.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)

                # Back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = pred.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels.to(device)).sum().item()

                # this line prints running loss and accuracy as the epoch progresses
                train_data_loader.set_postfix(loss=loss.item(), accuracy=100.0 * (correct / total))
                sleep(0.1)

            # this metric is not ideal, since it's the average loss over all batches.
            # We need the loss at the last iteration of data loader
            train_loss = running_loss / len(train_dataloader)
            train_loss_values.append(train_loss)
            accu = 100.0 * correct / total
            train_accuracy.append(accu)

        model.eval()
        with tqdm(test_loader, unit="batches") as test_data_loader:

            running_loss = 0.0
            correct = 0
            total = 0

            for imgs, labels in test_data_loader:

                test_data_loader.set_description(f"Testing: Epoch {epoch}/{epochs}")

                X, y = imgs.to(device), labels.to(device)
                pred = model(X)

                # compute test Loss
                loss = loss_fn(pred, y)
                running_loss += loss.item()
                _, predicted = torch.max(pred.data, 1)
                total += labels.size(0)

                # correct += (predicted == labels).sum().item()
                correct += predicted.eq(labels.to(device)).sum().item()

                # this line prints running loss and accuracy as the epoch progresses
                test_data_loader.set_postfix(loss=loss.item(), accuracy=100.0 * (correct / total))
                sleep(0.1)

            # this metric is not ideal, since it's the average loss over all batches.
            # We need the loss at the last iteration of data loader
            test_loss = running_loss / len(test_dataloader)
            test_loss_values.append(test_loss)
            accu = 100.0 * correct / total
            test_accuracy.append(accu)


def select_optimiser(argument, model):

    if argument == "sgd":
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    if argument == "sgd_nest":
        return optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
        )

    if argument == "adagrad":
        return optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if argument == "adadelta":
        return optim.Adadelta(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if argument == "adam":
        return optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


if __name__ == "__main__":

    """
    ARGUMENT PROVISION
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-en", "--experiment-number", type=str, required=True, help="number to track the different experiments"
    )
    parser.add_argument("-o", "--optimiser", type=str, required=True, help="optimizer for training")
    parser.add_argument("-d", "--device", type=str, required=False, default="gpu", help="device to train on")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=120, help="number of epochs to train for")
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        required=False,
        default=0.1,
        help="learning rate for the optimizer",
    )
    parser.add_argument(
        "-m", "--momentum", type=float, required=False, default=0.9, help="momentum value for optimizer if applicable"
    )
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        required=False,
        default=5e-4,
        help="weight decay value for the optimizer if applicable",
    )
    parser.add_argument("-dp", "--data-path", type=str, required=True, help="path to the dataset")
    parser.add_argument(
        "-wp",
        "--weight-path",
        type=str,
        required=False,
        default="_weights.pth",
        help="path to weights of the trained model",
    )
    parser.add_argument("-b", "--blocks", nargs=4, required=True, type=int, help="number of blocks in each layer")
    parser.add_argument("-c", "--channels", nargs=4, required=True, type=int, help="number of channels in each layer")
    args = parser.parse_args()

    """
    HYPERPARAMETERS
    """
    if args.device == "gpu" and torch.cuda.is_available() == True:
        device = "cuda"
    else:
        device = "cpu"

    data_path = args.data_path
    epochs = args.epochs
    blocks = args.blocks
    channels = args.channels

    resnet_model = project1_model(blocks, channels).to(device)
    optimizer = select_optimiser(args.optimiser, resnet_model)
    loss = nn.CrossEntropyLoss()

    """
    DATA RELATED STUFF
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    train_data = torchvision.datasets.CIFAR10(data_path, train=True, transform=train_transforms, download=True)
    test_data = torchvision.datasets.CIFAR10(data_path, train=False, transform=test_transforms, download=True)

    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    """
    MODEL TRAINING AND EVALUATION
    """
    print("Started Training\n")

    train_loss_values = []
    test_loss_values = []
    train_accuracy = []
    test_accuracy = []

    train_test_model(
        epochs, train_dataloader, test_dataloader, resnet_model, loss, optimizer, train_loss_values, train_accuracy
    )

    print("Finished Training\n")

    ## Save the train_loss_values and test_loss_values us np.save function

    np.save((args.experiment_number + "_train_loss_values.npy"), train_loss_values)
    np.save((args.experiment_number + "_test_loss_values.npy"), test_loss_values)
    np.save((args.experiment_number + "_train_accuracy.npy"), train_accuracy)
    np.save((args.experiment_number + "_test_accuracy.npy"), test_accuracy)

    PATH_MODEL_WEIGHTS = args.experiment_number + args.weight_path
    torch.save(resnet_model.state_dict(), PATH_MODEL_WEIGHTS)

    print("Model Saved\n")
    print(f"Final Training Accuracy: {train_accuracy[-1]: .3f}| Final Test Accuracy: {test_accuracy[-1]: .3f}\n")
    print("This accuracy is achieved with the following Settings")
    print(f"Number of Blocks in each layer: {blocks}\n Number of Channels in each layer: {channels}")
    print(f"Optimiser: {args.optimiser}\n Learning Rate: {args.learning_rate}")
