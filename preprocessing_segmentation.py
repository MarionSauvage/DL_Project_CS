""" 
File containg methods to preprocess images for classification task
"""

import os
import random
import albumentations as A
from albumentations.pytorch import ToTensor, ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from albumentations.pytorch import ToTensor, ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


PATCH_SIZE=128
#data augmentation

transforms = A.Compose([
    A.Resize(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    A.Normalize(p=1.0),
    ToTensor(),
])

#dataset preparation for training
class BrainMriDataset(Dataset):
    def __init__(self, df, transforms):
        
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, idx):
        """ 
        returns :
        - augmented image
        - mask
        - label 

        """        
        print("img",self.df.iloc[idx, 2])
        print("mask",self.df.iloc[idx, 3])
        print("other",self.df.iloc[idx, 1])
        image = cv2.imread(self.df.iloc[idx, 1])
        mask = cv2.imread(self.df.iloc[idx, 3], 0)
        #We take into account the mask
        augmented = self.transforms(image=image,mask=mask)
        
        return augmented

def get_train_test_val_sets(df,data_transforms=transforms):
    """Prepare dataset
    Keyword arguments:
    global_dataframe -- given by the "load_dataset" function
    Return: train, test, val sets loaders 
    """
    #df=global_dataframe[['image_path','tumor']]
    train_df, test_df = train_test_split(df, stratify=df.tumor, test_size=0.15, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Split train_df into train_df and test_df
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Train: {train_df.shape} \nVal: {val_df.shape} \nTest: {test_df.shape}")
    
    # train
    train_dataset = BrainMriDataset(df=train_df, transforms=data_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=26, num_workers=4, shuffle=True)

    # val
    val_dataset = BrainMriDataset(df=val_df, transforms=data_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=26, num_workers=4, shuffle=True)

    #test
    test_dataset = BrainMriDataset(df=test_df, transforms=data_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=26, num_workers=4, shuffle=True)
    return train_dataloader,test_dataloader,val_dataloader
