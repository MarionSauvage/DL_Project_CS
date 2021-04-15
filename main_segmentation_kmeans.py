from preprocessing import load_dataset
from segmentation_kmeans.preprocessing_segmentation import get_train_test_val_sets, get_k_splits_test_set
from segmentation_kmeans.model_segmentation import build_model
from segmentation_kmeans.segmentation import train_segmentation, evaluate_model, get_predictions_data, k_fold_cross_validation
from segmentation_kmeans.result_display import *
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
        optimizer = Adam(model.parameters(), lr=FLAGS.lr)
        criterion = BCELoss().cuda()

        # Train the model
        print("Training the model...")
        val_loss_history, val_dice_history, val_iou_history, val_pixel_acc_history = train_segmentation(model=model, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=FLAGS.epochs, early_stop=FLAGS.early_stopping)
        
        # Print loss, dice and iou history
        print("Dice history (val): ", val_dice_history)
        print("IoU history (val): ", val_iou_history)
        print("Pixel accuracy history (val): ", val_iou_history)
        print("Loss history (val): ", val_loss_history)

        # Performance evaluation on test data
        avg_loss_test, dice_test, iou_test, pixel_acc_test = evaluate_model(model, device, test_loader, criterion)
        print("Dice (test): {:.1%}".format(dice_test))
        print("IoU (test): {:.1%}".format(iou_test))
        print("Pixel accuracy (test): {:.1%}".format(pixel_acc_test))
        print("Loss (test): {:1.4f}".format(avg_loss_test))

        if FLAGS.display_predictions:
            predictions = get_predictions_data(model, device, test_loader)
            display_predictions(predictions)

        # Save model paramters to disk if save path is provided
        if FLAGS.save_model:
            print("Saving model...")
            model.to('cpu')
            torch.save(model.state_dict(), './models/' + FLAGS.model + '_model.pt')

    elif FLAGS.mode == 'learning_rate_comparison':
        lr_list = [1e-4]
        epochs = [i for i in range(FLAGS.epochs)]

        for lr in lr_list:
            model = build_model(FLAGS.model)

            # defining the optimizer and loss function
            print("Learning rate: ", lr)
            optimizer = Adam(model.parameters(), lr=lr)
            criterion = BCELoss().cuda()

            # Train the model
            print("Training the model...")
            val_loss_history, val_dice_history, val_iou_history, val_pixel_acc_history = train_segmentation(model=model, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=FLAGS.epochs, early_stop=FLAGS.early_stopping)
            
            # Print loss, dice and iou history
            print("Dice history (val): ", val_dice_history)
            print("IoU history (val): ", val_iou_history)
            print("Pixel accuracy history (val): ", val_pixel_acc_history)
            print("Loss history (val): ", val_loss_history)

            # Save validation loss graph
            plt.plot(epochs, val_loss_history)
            plt.title(f'Validation loss, lr = {lr}')
            plt.savefig(f'./graphs/{FLAGS.model}-val_loss_lr_{lr}.png')
            plt.clf()

            # Save validation dice graph
            plt.plot(epochs, val_dice_history)
            plt.title(f'Validation dice, lr = {lr}')
            plt.savefig(f'./graphs/{FLAGS.model}-val_dice_lr_{lr}.png')
            plt.clf()

            # Save validation IoU graph
            plt.plot(epochs, val_iou_history)
            plt.title(f'Validation IoU, lr = {lr}')
            plt.savefig(f'./graphs/{FLAGS.model}-val_iou_lr_{lr}.png')
            plt.clf()

            # Save validation pixel accuracy graph
            plt.plot(epochs, val_pixel_acc_history)
            plt.title(f'Validation pixel accuracy, lr = {lr}')
            plt.savefig(f'./graphs/{FLAGS.model}-val_pixel_acc_lr_{lr}.png')
            plt.clf()

            # Performance evaluation on test data
            avg_loss_test, dice_test, iou_test, pixel_acc_test = evaluate_model(model, device, test_loader, criterion)
            print("Dice (test): {:.1%}".format(dice_test))
            print("IoU (test): {:.1%}".format(iou_test))
            print("Pixel accuracy (test): {:.1%}".format(pixel_acc_test))
            print("Loss (test): {:1.4f}".format(avg_loss_test))
    
    elif FLAGS.mode == 'batch_size_comparison':
        batch_list = [20, 40, 55, 70]
        epochs = [i for i in range(FLAGS.epochs)]

        # Model initialization
        model = build_model(FLAGS.model)
        optimizer = Adam(model.parameters(), lr=FLAGS.lr)
        criterion = BCELoss().cuda()

        for batch_size in batch_list:
            # Get proper dataloader for the given batch_size
            print("Batch size: ", batch_size)
            train_loader,test_loader,val_loader=get_train_test_val_sets(dataset, batch_size=batch_size)

            # Train the model
            print("Training the model...")
            val_loss_history, val_dice_history, val_iou_history, val_pixel_acc_history = train_segmentation(model=model, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=FLAGS.epochs, early_stop=FLAGS.early_stopping)

            # Print loss, dice and iou history
            print("Dice history (val): ", val_dice_history)
            print("IoU history (val): ", val_iou_history)
            print("Pixel accuracy history (val): ", val_pixel_acc_history)
            print("Loss history (val): ", val_loss_history)


            # Performance evaluation on test data
            avg_loss_test, dice_test, iou_test, pixel_acc_test = evaluate_model(model, device, test_loader, criterion)
            print("Dice (test): {:.1%}".format(dice_test))
            print("IoU (test): {:.1%}".format(iou_test))
            print("Pixel accuracy (test): {:.1%}".format(pixel_acc_test))
            print("Loss (test): {:1.4f}".format(avg_loss_test))

    elif FLAGS.mode == 'k_fold_cross_validation':
        # Get split indices and test set loader
        split_indices, test_loader = get_k_splits_test_set(dataset, FLAGS.nb_splits)

        # Perform k-fold cross validation
        k_fold_cross_validation(dataset, split_indices, FLAGS.model, device, FLAGS.lr, epochs=FLAGS.epochs, early_stop=FLAGS.early_stopping)

if __name__ == '__main__':
    # Command line arguments setup
    FLAGS = flags.FLAGS
    flags.DEFINE_enum('mode', 'basic', ['basic', 'learning_rate_comparison', 'batch_size_comparison', 'k_fold_cross_validation'], '')
    flags.DEFINE_enum('model', 'Unet', ['Unet', 'UnetResNet'], '')
    flags.DEFINE_integer('epochs', 50, '')
    flags.DEFINE_integer('early_stopping', 7, '')
    flags.DEFINE_integer('nb_splits', 5, '')
    flags.DEFINE_float('lr', 1e-4, '')
    flags.DEFINE_bool('display_predictions', False, '')
    flags.DEFINE_bool('save_model', False, '')

    app.run(main)
