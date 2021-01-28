from preprocessing import load_dataset
from preprocessing_segmentation import get_train_test_val_sets
from segmentation import build_model

DATA_PATH = "../kaggle_3m/"
dataset=load_dataset(DATA_PATH)

train_loader,test_loader,val_loader=get_train_test_val_sets(dataset)

unet = build_model()
print(unet)


