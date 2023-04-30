'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html
On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
On Tensors
----------
LinearTransformation, Normalize, RandomErasing
Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda 
Custom
------
Write own class
Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision

from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class WineDataset(Dataset):
    def __init__(self,transform=None):
        xy = pd.read_csv(
            "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv").to_numpy(
            dtype=np.float32)
        self.x = xy[:, 1:]  # size [n_samples, n_features]
        self.y = xy[:, [0]]  # size [n_samples, 1]
        self.no_samples = xy.shape[0]
        self.no_features = xy.shape[1]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index],self.y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.no_samples


# Custom Transforms
# implement __call__(self, sample)
class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MultiTransform:
    def __init__(self,factor):
        self.factor = factor

    def __call__(self, sample):
        input, target = sample
        input = self.factor * input
        return input, target


print('Without Transform')
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MultiTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)