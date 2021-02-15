
from preprocessing import load_dataset
from preprocessing_segmentation import get_train_test_val_sets
from model_segmentation import build_model
from segmentation import train_segmentation,evaluate_model
import torch
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, BCELoss
from absl import app, flags
import matplotlib.pyplot as plt


def main(argv):
    # Set torch seed
    torch.manual_seed(42)

    #data import
    DATA_PATH="../dataset_mri/lgg-mri-segmentation/kaggle_3m/"
    dataset=load_dataset(DATA_PATH)
    print(dataset.head())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #dataset splitted 
    train_loader,test_loader,val_loader=get_train_test_val_sets(dataset)
    print(train_loader)

    if FLAGS.mode == 'basic':
        model = build_model(FLAGS.model)
        print(model)


        # defining the optimizer and loss function
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = BCELoss().cuda()

        # Train the model
        print("Training the model...")
        val_loss_history, val_iou_history = train_segmentation(model=model, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=50)

        # Performance evaluation on test data
        avg_loss_test, iou_test = evaluate_model(model, device, test_loader, optimizer, criterion)
        print("IoU (test): {:.1%}".format(iou_test))
        print("Loss (test): {:1.4f}".format(avg_loss_test))

    elif FLAGS.mode == 'learning_rate_comparison':
        lr_list = [5e-5]
        epochs = [i for i in range(FLAGS.epochs)]

        for lr in lr_list:
            model = build_model(FLAGS.model)

            # defining the optimizer and loss function
            print("Learning rate: ", lr)
            optimizer = Adam(model.parameters(), lr=lr)
            criterion = BCELoss().cuda()

            # Train the model
            print("Training the model...")
            val_loss_history, val_iou_history = train_segmentation(model=model, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=FLAGS.epochs)
            
            # Save validation loss graph
            plt.plot(epochs, val_loss_history)
            plt.title(f'Validation loss, lr = {lr}')
            plt.savefig(f'./graphs/val_loss_lr_{lr}.png')
            plt.clf()

            # Save validation IoU graph
            plt.plot(epochs, val_iou_history)
            plt.title(f'Validation IoU, lr = {lr}')
            plt.savefig(f'./graphs/val_iou_lr_{lr}.png')
            plt.clf()

            # Performance evaluation on test data
            avg_loss_test, iou_test = evaluate_model(model, device, test_loader, optimizer, criterion)
            print("IoU (test): {:.1%}".format(iou_test))
            print("Loss (test): {:1.4f}".format(avg_loss_test))

if __name__ == '__main__':
    # Command line arguments setup
    FLAGS = flags.FLAGS
    flags.DEFINE_enum('mode', 'basic', ['basic', 'learning_rate_comparison'], '')
    flags.DEFINE_enum('model', 'Unet', ['Unet', 'UnetResNet'], '')
    flags.DEFINE_integer('epochs', 50, "")

    app.run(main)