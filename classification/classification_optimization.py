import os
import torch
from torch.autograd import Variable
from ray import tune
from preprocessing import load_dataset
from classification.preprocessing_for_classification import *
from classification.classification import *
from classification.model_classifier import *

# Path to all data
DATA_PATH = "../dataset_mri/lgg-mri-segmentation/kaggle_3m/"

# Load dataset
dataset = load_dataset(DATA_PATH)

def hyperparam_optimizer(config):
    # Switch back to current dir instead of the custom one set by tune
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Initialize model, datasets and criterion
    train_dataloader, test_dataloader, val_dataloader = get_train_test_val_sets(dataset)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(4, 3, 10)

    criterion = CrossEntropyLoss()

    # Initialize the optimizer
    optimizer = SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    while True:
        accuracy, f1_score = train_classification(model, device, train_dataloader, val_dataloader, optimizer, criterion, epochs=20)

        # Run tune
        tune.report(avg_accuracy=accuracy)


def hyperparam_nn_layers(config):
    # Switch back to current dir instead of the custom one set by tune
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Initialize datasets
    train_dataloader, test_dataloader, val_dataloader = get_train_test_val_sets(dataset)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use hyperparameters provided by tune
    model = build_model(config['conv_out_features'], config['conv_kernel_size'], config['linear_features'])

    # Initialize the optimizer and criterion
    optimizer = SGD(model.parameters(), lr=0.00095, momentum=0.874)
    criterion = CrossEntropyLoss()


    while True:
        accuracy, f1_score = train_classification(model, device, train_dataloader, val_dataloader, optimizer, criterion, epochs=20)

        # Run tune
        tune.report(avg_accuracy=accuracy)