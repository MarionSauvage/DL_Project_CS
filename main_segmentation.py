#from preprocessing_segmentation import *
from segmentation import Unet,DoubleConv



unet = UNet(n_classes=1).to(device)
output = unet(torch.randn(1,3,256,256).to(device))
print("",output.shape)