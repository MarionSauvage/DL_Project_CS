from preprocessing import load_dataset
from preprocessing_segmentation import get_train_test_val_sets
from segmentation import build_model
from classification import train_classification,evaluate_model

#data import
DATA_PATH = "../kaggle_3m/"
dataset=load_dataset(DATA_PATH)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#dataset splitted 
train_loader,test_loader,val_loader=get_train_test_val_sets(dataset)

unet = build_model()
print(unet)


# defining the optimizer and loss function
optimizer = SGD(model.parameters(), lr=0.05)
criterion = CrossEntropyLoss()

# Train the model
print("Training the model...")
train_classification(model, device, train_loader, val_loader, optimizer, criterion, epochs=5)

# Performance evaluation on test data
loss, accuracy = evaluate_model(model, device, test_loader, optimizer, criterion)
print("Accuracy (test): {:.1%}".format(accuracy))