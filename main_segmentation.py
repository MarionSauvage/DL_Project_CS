
from preprocessing import load_dataset
from preprocessing_segmentation import get_train_test_val_sets
from model_segmentation import build_model
from segmentation import train_segmentation,evaluate_model
import torch
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, BCELoss

if __name__=='__main__':
    #data import
    DATA_PATH="../dataset_mri/lgg-mri-segmentation/kaggle_3m/"
    dataset=load_dataset(DATA_PATH)
    print(dataset.head())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #dataset splitted 
    train_loader,test_loader,val_loader=get_train_test_val_sets(dataset)
    print(train_loader)

    unet = build_model()
    print(unet)


    # defining the optimizer and loss function
    optimizer = Adam(unet.parameters(), lr=1e-3)
    criterion = BCELoss().cuda()

    # Train the model
    print("Training the model...")
    loss_history, iou_train_history, val_iou_history=train_segmentation(model=unet, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=4)

    # Performance evaluation on test data
    avg_loss_test, iou_test = evaluate_model(unet, device, test_loader, optimizer, criterion)
    print("IoU (test): {:.1%}".format(iou_test))
    print("Loss (test): {:1.4f}".format(avg_loss_test))
