import pandas as pd 
import os
import numpy as np 
import random
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from ray import tune
from preprocessing import load_dataset
from preprocessing_for_classification import *
from model_classifier import *
from classification import *

# Path to all data
DATA_PATH = "../lgg-mri-segmentation/kaggle_3m/"

# Load dataset
dataset = load_dataset(DATA_PATH)

# Separate dataset into train, validation and test dataset
# print("Size of the datasets:")
# train_loader, test_loader, val_loader = get_train_test_val_sets(dataset)
# print("\n")


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = build_model(4, 3, 10)


# # defining the optimizer and loss function
# optimizer = SGD(model.parameters(), lr=0.05)
# criterion = CrossEntropyLoss()

# # Train the model
# print("Training the model...")
# train_classification(model, device, train_loader, val_loader, optimizer, criterion, epochs=20)

# # Performance evaluation on test data
# loss, accuracy = evaluate_model(model, device, test_loader, optimizer, criterion)
# print("Accuracy (test): {:.1%}".format(accuracy))

# Hyperparameters evaluation
optimizer_analysis = tune.run(
        hyperparam_optimizer,
        metric="avg_accuracy",
        mode="max",
        stop={
            "mean_accuracy": 0.98,
            "training_iteration": 5
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        },
        num_samples=50,
        config={
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        })