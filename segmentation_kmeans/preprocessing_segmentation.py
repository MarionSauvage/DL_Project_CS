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
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


PATCH_SIZE=128 

# Data augmentation
data_aug_transforms = A.Compose([
    A.Resize(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    A.Normalize(p=1.0),
    ToTensor(),
])

test_transforms = A.Compose([
    A.Resize(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
    A.Normalize(p=1.0),
    ToTensor(),
])

# Dataset preparation for training
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
        image = cv2.imread(self.df.iloc[idx, 3])
        mask = cv2.imread(self.df.iloc[idx, 1], 0)
        #We take into account the mask
        augmented = self.transforms(image=image,mask=mask)
        
        return augmented

def get_train_test_val_sets(df, data_aug_transforms=data_aug_transforms, test_transforms=test_transforms, batch_size=20):
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
    train_dataset = BrainMriDataset(df=train_df, transforms=data_aug_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # val
    val_dataset = BrainMriDataset(df=val_df, transforms=data_aug_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    #test
    test_dataset = BrainMriDataset(df=test_df, transforms=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    return train_dataloader,test_dataloader,val_dataloader

def get_k_splits_test_set(df, n_splits=5, test_transforms=test_transforms):
    """ Prepare dataset for k-fold cross-validation
    Keyword arguments:
        df                  -- given by the "load_dataset" function
        n_splits            -- number of folds to split the dataset
        test_transforms     -- transforms object for the test dataloader
    Return: k-folds indices, test set loader 
    """

    # Split df into train+val dataset and test_df
    train_val_df, test_df = train_test_split(df, stratify=df.tumor, test_size=0.15, random_state=42)
    train_val_df = train_val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Get split indices for k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    split_indices = kf.split(train_val_df)

    # Get test dataloader
    test_dataset = BrainMriDataset(df=test_df, transforms=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=20, num_workers=4, shuffle=True)

    return split_indices, test_dataloader


def get_train_val_splits_set(df, train_indices, val_indices, data_aug_transforms=data_aug_transforms):
    """ Get train and validation loaders
    Keyword arguments:
        df                  -- given by the "load_dataset" function
        train_indices       -- indices for a given train split
        val_indices         -- indices for the corresponding validation split
        data_aug_transforms -- transforms object for the train and validation dataloaders
    Return: train, val sets loaders 
    """
    # Get train and validation datasets
    train_df = df.iloc[list(train_indices)]
    val_df = df.iloc[list(val_indices)]
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Get train set loader
    train_dataset = BrainMriDataset(df=train_df, transforms=data_aug_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=20, num_workers=4, shuffle=True)

    # Get validation set loader
    val_dataset = BrainMriDataset(df=val_df, transforms=data_aug_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=20, num_workers=4, shuffle=True)

    return train_dataloader, val_dataloader