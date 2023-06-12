# 1) prepare the dataset
# 2) write the forward pass function to compute prediction
# 3) define the loss function
# 4) run through the training loop to minimize the loss function
# 5) run the train model with test data


# End to end deep learning example without torch


# 1) Prepare the dataset

import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=np.float32)

w = 0.0


# 2) Write forward pass function -> y = 2x
def forward_pass(x):
    return w * x


# 3) loss function > loss = MSE (Mean square error)

# MSE = 1/N * (y_prediction - y )**2

def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()


# 4) define the gradient FUNCTION

# in this case, gradient of MSE --> dJ/dw =  d (1/N ) * (y_pred -y)**2/ dw

#                                   dJ/dw = d (1/N) * (wx-y)**2/dw
#                                   dJ/dw = 1/N * 2 * (wx-y) * d(wx-y)/dw
#                                   dJ/dw = 1/N * 2 * (mx-y) * x
#                                   dJ/dw = 1/N * 2x * (mx-y)

def gradient(x,y,y_predicted):
    return (2*x*(y_predicted-y)).mean()

print(f'Prediction before training: f(5) = {forward_pass(5):.3f}')



# 5) Training

learning_rate = 0.01
n_iter = 100

# training loop


for epoch in range(n_iter):
    # prediction = forward_pass
    y_pred = forward_pass(x)

    #loss

    l = loss(y,y_pred)

    # gradients

    dw = gradient(x,y,y_pred)

    # update weights

    w -= learning_rate* dw

    if epoch % 5 ==0:
        print(f"epoch : {epoch+1} , weights = {w:.3f} , loss = {l:3f},  ")


print(f'Prediction after training: f(5) = {forward_pass(5):.3f}')


#####################################################################################

#  End to end deep learning example without torch


import torch
x_torch = torch.tensor([1,2,3,4,5,6,7,8,9,10],dtype=torch.float32)
y_torch = torch.tensor([2,4,6,8,10,12,14,16,18,20],dtype=torch.float32)
w=0.0
no_iters_torch = 100

learning_rate_torch = 0.01

print(f"Before prediction :f(5) = {forward_pass(x_torch)}")

for epoch in range(no_iters_torch):
    y_pred_torch = forward_pass(x_torch)

    l_torch = loss(y_torch,y_pred_torch)

    dw_torch = gradient(x_torch,y_torch,y_pred_torch)

    w -= learning_rate_torch*dw_torch

    if epoch % 5 == 0:
        print(f"epoch : {epoch + 1} , weights = {w:.3f} , loss = {l:3f},  ")

print(f'Prediction after training: f(5) = {forward_pass(5):.3f}')



