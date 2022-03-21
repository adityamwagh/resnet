import argparse
import torch
import torchvision
import torchvision.transforms as transforms

train_transforms1 = transforms.Compose(
        [
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

train_transforms2 = transforms.Compose(
    [
        transforms.RandomRotation(30),
        transforms.RandomCrop((32,32), padding=4)
        transforms.RandomHorizontalFlip(p=0.5),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

train_transforms3 = transforms.Compose([
data_path  = './data/'

train_data = torchvision.datasets.CIFAR10(data_path, train=True, transform=train_transforms, download=False)

print(train_data[0][0].shape)