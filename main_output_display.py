import torch
from torch.nn import BCELoss
from segmentation.result_display import display_predictions, display_predictions_20
from segmentation.model_segmentation import build_model
from preprocessing import load_dataset
from segmentation.preprocessing_segmentation import get_train_test_val_sets
from segmentation.segmentation import  get_predictions_data
from absl import app, flags
import matplotlib.pyplot as plt

def main(argv):
    torch.manual_seed(42)
    if FLAGS.model == 'Unet':
        PATH="models/Unet_model.pt"
    elif  FLAGS.model == 'UnetResNet':
        PATH="models/UnetResNet_model.pt"

    #data import
    DATA_PATH="../dataset_mri/lgg-mri-segmentation/kaggle_3m/"
    dataset=load_dataset(DATA_PATH)
    print(dataset.head())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #dataset splitted 
    train_loader,test_loader,val_loader=get_train_test_val_sets(dataset)
    print(train_loader)

    model = build_model(FLAGS.model)
    print(model)

    # Load saved model
    model.load_state_dict(torch.load(PATH), strict=False)
    model.eval()

    # Performance evaluation on test data
    criterion = BCELoss().cuda()
    avg_loss_test, dice_test, iou_test, pixel_acc_test = evaluate_model(model, device, test_loader, criterion)
    print("Dice (test): {:.1%}".format(dice_test))
    print("IoU (test): {:.1%}".format(iou_test))
    print("Pixel accuracy (test): {:.1%}".format(pixel_acc_test))
    print("Loss (test): {:1.4f}".format(avg_loss_test))

    predictions = get_predictions_data(model, device, test_loader)
    display_predictions_20(predictions)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_enum('model', 'Unet', ['Unet', 'UnetResNet'], '')
    app.run(main)
