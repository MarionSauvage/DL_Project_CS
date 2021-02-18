from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from ray import tune
from absl import app, flags
from preprocessing import load_dataset
from preprocessing_for_classification import *
from model_classifier import *
from classification import *
from classification_optimization import *

def main(argv):
    if FLAGS.mode == 'basic':
        #### Basic training and evaluation of the classification model ####
        # Path to all data
        DATA_PATH = "../dataset_mri/lgg-mri-segmentation/kaggle_3m/"

        # Load dataset
        dataset = load_dataset(DATA_PATH)

        # Separate dataset into train, validation and test dataset
        print("Size of the datasets:")
        train_loader, test_loader, val_loader = get_train_test_val_sets(dataset)
        print("\n")


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = build_model(6, 4, 20)


        # defining the optimizer and loss function
        optimizer = SGD(model.parameters(), lr=0.00095, momentum=0.874)
        criterion = CrossEntropyLoss()

        # Train the model
        print("Training the model...")
        train_classification(model, device, train_loader, val_loader, optimizer, criterion, epochs=20)

        # Performance evaluation on test data
        loss, accuracy = evaluate_model(model, device, test_loader, optimizer, criterion)
        print("Accuracy (test): {:.1%}".format(accuracy))

    #### Hyperparameters selection ####
    elif FLAGS.mode == 'optimizer_optimization':
        # Learning rate and momentum
        optimizer_analysis = tune.run(
                hyperparam_optimizer,
                metric="avg_accuracy",
                mode="max",
                stop={
                    "avg_accuracy": 0.98,
                    "training_iteration": 1
                },
                resources_per_trial={
                    "cpu": 3,
                    "gpu": 0.33
                },
                num_samples=30,
                config={
                    "lr": tune.loguniform(1e-4, 1e-2),
                    "momentum": tune.uniform(0.1, 0.9),
                })
    elif FLAGS.mode == 'nn_layers_optimization':
        # neural network layers optimization
        optimizer_analysis = tune.run(
                hyperparam_nn_layers,
                metric="avg_accuracy",
                mode="max",
                stop={
                    "avg_accuracy": 0.98,
                    "training_iteration": 2
                },
                resources_per_trial={
                    "cpu": 3,
                    "gpu": 0.33
                },
                config={
                    "conv_out_features": tune.grid_search([4, 5, 6]),
                    "conv_kernel_size": tune.grid_search([3, 4, 5]),
                    "linear_features": tune.grid_search([5, 10, 15, 20]),
                })

if __name__ == '__main__':
    # Command line arguments setup
    FLAGS = flags.FLAGS
    flags.DEFINE_enum('mode', 'basic', ['basic', 'optimizer_optimization', 'nn_layers_optimization'], '')

    app.run(main)