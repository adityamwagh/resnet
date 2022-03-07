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
def train_model(epochs, loader, model, loss_fn, optimizer, train_loss_values, train_accuracy):

    model.train()
    epoch_time = 0.0
    for epoch in range(1, epochs + 1):
        with tqdm(loader, unit="batches") as data_loader:

            running_loss = 0
            correct = 0
            total = 0

            start = time.time()
            for imgs, labels in data_loader:

                data_loader.set_description(f"Epoch {epoch}/{epochs}")

                end = time.time()

                X, y = imgs.to(device), labels.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)

                # Back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                end = time.time()

                running_loss += loss.item()
                _, predicted = pred.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels.to(device)).sum().item()

                # this line prints running loss and accuracy as the epoch progresses
                data_loader.set_postfix(loss=loss.item(), accuracy=100.0 * (correct / total))
                sleep(0.1)

            # this metric is not ideal, since it's the average loss over all batches.
            # We need the loss at the last iteration of data loader
            train_loss = running_loss / len(train_dataloader)
            train_loss_values.append(train_loss)
            accu = 100.0 * correct / total
            train_accuracy.append(accu)

            np.save("train_loss", train_loss_values)
            np.save("train_accuracy", train_accuracy)

            epoch_time = end - start
            print(f"time/epoch: {epoch_time: .3f}\n")


# Defining the Testing Loop
def test_model(epochs, loader, model, loss, test_loss_values, test_accuracy):

    model.eval()
    for epoch in range(1, epochs + 1):
        with tqdm(loader, unit="batches") as data_loader:

            running_loss = 0.0
            correct = 0
            total = 0

            for imgs, labels in data_loader:

                data_loader.set_description(f"Epoch {epoch}/{epochs}")

                X, y = imgs.to(device), labels.to(device)
                pred = model(X)

                # compute test Loss
                fit = loss(pred, y)
                running_loss += fit.item()
                _, predicted = torch.max(pred.data, 1)
                total += labels.size(0)

                # correct += (predicted == labels).sum().item()
                correct += predicted.eq(labels.to(device)).sum().item()

                # this line prints running loss and accuracy as the epoch progresses
                data_loader.set_postfix(loss=loss.item(), accuracy=100.0 * (correct / total))
                sleep(0.1)

            # this metric is not ideal, since it's the average loss over all batches.
            # We need the loss at the last iteration of data loader
            test_loss = running_loss / len(test_dataloader)
            test_loss_values.append(test_loss)
            accu = 100.0 * correct / total
            test_accuracy.append(accu)

            np.save("test_loss", test_loss_values)
            np.save("test_accuracy", test_accuracy)


def select_optimiser(argument, model):

    if argument == "sgd":
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    if argument == "sgd_nest":
        return optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
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
    parser.add_argument("-w", "--workers", type=int, required=True, help="number of workers for the data loader")
    parser.add_argument("-o", "--optimiser", type=str, required=True, help="optimizer for training")
    parser.add_argument("-d", "--device", type=str, required=True, help="device to train on")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="number of epochs to train for")
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        required=False,
        default=0.01,
        help="learning rate for the optimizer",
    )
    parser.add_argument(
        "-m",
        "--momentum",
        type=float,
        required=False,
        default=0.9,
        help="momentum value for optimizer if applicable",
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
    parser.add_argument("-wp", "--weight-path", type=str, required=True, help="path to weights of the trained model")
    args = parser.parse_args()

    """
    HYPERPARAMETERS
    """
    num_workers = args.workers
    data_path = args.data_path
    if args.device == "gpu" and torch.cuda.is_available() == True:
        device = "cuda"
    else:
        device = "cpu"
    epochs = args.epochs
    resnet_model = project1_model().to(device)
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

    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=num_workers)

    """
    MODEL TRAINING AND EVALUATION
    """
    print("Started Training")

    train_loss_values = []
    test_loss_values = []
    train_accuracy = []
    test_accuracy = []

    train_model(epochs, train_dataloader, resnet_model, loss, optimizer, train_loss_values, train_accuracy)
    test_model(epochs, test_dataloader, resnet_model, loss, test_loss_values, test_accuracy)

    print("Finished Training")

    PATH_MODEL_WEIGHTS = args.weight_path
    torch.save(resnet_model.state_dict(), PATH_MODEL_WEIGHTS)

    print("Model Saved")
