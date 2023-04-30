import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import pandas as pd


class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = pd.read_csv(
            "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv").to_numpy(
            dtype=np.float32)
        self.x = torch.from_numpy(xy[:, 1:])  # size [n_samples, n_features]
        self.y = torch.from_numpy(xy[:, [0]])  # size [n_samples, 1]
        self.no_samples = xy.shape[0]
        self.no_features = xy.shape[1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.no_samples


dataset = WineDataset()
print(dataset.no_features)
print(len(dataset))
features, label = dataset[0]

print(features)
print(label)

# Dataloader
# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses

train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)

# convert to an iterator and look at one random sample
dataiter = iter(train_loader)
data = next(dataiter)
features, labels = data
print("data loader")
print(features, labels)

# trainin loop

num_epoch = 2
no_samples = len(dataset)
no_iterations = math.ceil(no_samples/4)

print(no_iterations)

for epoch in range(num_epoch):
    for i, (inputs,labels) in enumerate(train_loader):
        # forward backward + update weights

        if (i+1)%5==0:
            print(f"epoch: {epoch+1}/{num_epoch} , step = {i+1}/{no_iterations}, input {inputs.shape}")




### torchvision  - already datasets
# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

train_dataset_MNIST = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)


print(train_dataset_MNIST)

train_loader = DataLoader(dataset=train_dataset_MNIST,
                                           batch_size=1,
                                           shuffle=True)

# look at one random sample
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data

print(inputs)
print(inputs.shape, targets.shape)

