import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torchvision import models
from torchvision.models import resnext50_32x4d, resnet18

resnet_backbone = resnet18(pretrained=True)

def DoubleConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))

class Unet(nn.Module):
    def __init__(self,nb_classes=1):
        super().__init__()
        #Left side of UNET : Sequential NN
        self.conv_left1 = DoubleConv(3,64)
        self.conv_left2 = DoubleConv(64, 128)
        self.conv_left3 = DoubleConv(128, 256)
        self.conv_left4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #Right side of UNET :
        self.conv_right3 = DoubleConv(256 + 512, 256)
        self.conv_right2 = DoubleConv(128 + 256, 128)
        self.conv_right1 = DoubleConv(128 + 64, 64)

        self.last_conv = nn.Conv2d(64, nb_classes, kernel_size=1)

    def forward(self,x):
        conv1 = self.conv_left1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_left2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_left3(x)
        x = self.maxpool(conv3)
        conv4 = self.conv_left4(x)
        x = self.upsample(conv4)

        x = torch.cat([x, conv3], dim=1)

        x = self.conv_right3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.conv_right2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.conv_right1(x)

        out = self.last_conv(x)
        out = torch.sigmoid(out)
        return out

class UnetResNet(nn.Module):
    def __init__(self,nb_classes=1):
        super().__init__()

        ## Resnet backbone 
        self.base_model = resnet_backbone
        self.base_layers = list(resnet_backbone.children())

        #Left side of UNET : Sequential NN
        self.conv_left1 = nn.Sequential(self.base_layers[:4][0])
        self.conv_left2 = nn.Sequential(self.base_layers[5][0])
        self.conv_left3 = nn.Sequential(self.base_layers[6][0])
        self.conv_left4 = nn.Sequential(self.base_layers[7][0])

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        #Right side of UNET : 
        self.conv_right3 = DoubleConv(256 + 512, 256)
        self.conv_right2 = DoubleConv(128 + 256, 128)
        self.conv_right1 = DoubleConv(128 + 64, 64)
        
        self.last_conv = nn.Conv2d(64, nb_classes, kernel_size=1)

    def forward(self,x):
        conv1 = self.conv_left1(x)
        conv2 = self.conv_left2(conv1)
        conv3 = self.conv_left3(conv2)
        conv4 = self.conv_left4(conv3)
        x = self.upsample(conv4)

        x = torch.cat([x, conv3], dim=1)

        x = self.conv_right3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.conv_right2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.conv_right1(x)
        x = self.upsample(x)

        out = self.last_conv(x)
        out = torch.sigmoid(out)
        return out

def build_model(nn_type='Unet'):
    if nn_type == 'UnetResNet':
        model = UnetResNet()
    elif nn_type == 'UnetResNext':
        model = UnetResNext()
    else:
        model = Unet()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Puts model on GPU/CPU
    model.to(device)
    return model