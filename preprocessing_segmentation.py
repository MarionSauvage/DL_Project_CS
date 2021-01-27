

class BrainMriDataset(Dataset):
    def __init__(self, df, transforms):
        
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 1])
        mask = cv2.imread(self.df.iloc[idx, 3], 0)
        
        augmented_img = self.transforms(image)
        
        return augmented_img, self.df.iloc[idx, 5]

""" 
File containg methods to preprocess images for classification task
"""

from PIL import Image
import pandas as pd 
import os
import numpy as np 
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid import ImageGrid
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn

class BrainMriDataset(Dataset):
    def __init__(self, df, transforms):
        
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 1])
        mask = cv2.imread(self.df.iloc[idx, 3], 0)
        
        augmented_img = self.transforms(image)
        
        return augmented_img, self.df.iloc[idx, 5]

def get_train_test_val_sets(df):
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
    
    data_transforms = transforms.Compose([
        # transforms.Resize([224, 224]),
        transforms.ToTensor(),
        #transforms.Normalize(mean=mean, std=std)
    ])
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