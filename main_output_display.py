import torch
from segmentation.result_display import display_predictions
from segmentation.model_segmentation import build_model
from preprocessing import load_dataset
from segmentation.preprocessing_segmentation import get_train_test_val_sets
from segmentation.segmentation import  get_predictions_data
from absl import app, flags
import matplotlib.pyplot as plt

PATH="C:/Users/mario/Documents/cpu_unetresnet_model.pt"

def main(argv):
    torch.manual_seed(42)
    if FLAGS.model == 'Unet':
        PATH="models/unet_model.pt"
    elif  FLAGS.model == 'UnetResNet': 
        PATH="models/unetresnet_model.pt"

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
    model.load_state_dict(torch.load(PATH), strict=False)
    model.eval()
    predictions = get_predictions_data(model, device, test_loader)
    display_predictions(predictions)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_enum('model', 'Unet', ['Unet', 'UnetResNet'], '')
    app.run(main)
