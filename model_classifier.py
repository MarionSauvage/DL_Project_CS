import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

class Net(Module):   
    def __init__(self, conv_out_features, conv_kernel_size, linear_features):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, conv_out_features, kernel_size=conv_kernel_size, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            BatchNorm2d(conv_out_features),
            # Defining another 2D convolution layer
            Conv2d(conv_out_features, conv_out_features, kernel_size=conv_kernel_size, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            BatchNorm2d(conv_out_features),
        )

        self.linear_layers = Sequential(
            Linear(self.get_conv_out_size(), linear_features),
            ReLU(inplace=True),
            Linear(linear_features, 2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def get_conv_out_size(self):
        conv1_out_shape = self.cnn_layers[0](torch.rand(*(1, 3, 256, 256))).data.shape
        maxpool1_out_shape = self.cnn_layers[2](torch.rand(*(conv1_out_shape))).data.shape
        conv2_out_shape = self.cnn_layers[4](torch.rand(*(maxpool1_out_shape))).data.shape
        maxpool2_out_shape = self.cnn_layers[6](torch.rand(*(conv2_out_shape))).data.shape

        return np.prod(list(maxpool2_out_shape))

def build_model(conv_out_features, conv_kernel_size, linear_features):
    model = Net(conv_out_features, conv_kernel_size, linear_features)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Puts model on GPU/CPU
    model.to(device)
    return model

def reset_parameters_model(model):
    for layers in model.children():
        for layer in layers.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
