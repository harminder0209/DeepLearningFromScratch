import torch


import numpy as np


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


x = np.array([1.3,2.0,0.1])

output = softmax(x)

print(output)