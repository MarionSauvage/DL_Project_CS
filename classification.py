import numpy as np
import torch
from torch.autograd import Variable
from ray import tune
from preprocessing_for_classification import *


def evaluate_model(model, device, dataloader, optimizer, criterion):
    avg_accuracy = 0
    avg_loss = 0.0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)

            loss = criterion(out, targets)
            _, predictions = torch.max(out, 1)
            nb_correct = torch.sum(predictions == targets)

            avg_loss += loss.item()
            avg_accuracy += nb_correct
    
    return avg_loss / len(dataloader.dataset), float(avg_accuracy) / len(dataloader.dataset)


def train_classification(model, device, train_dataloader, val_dataloader, optimizer, criterion, epochs=20):
    # Early stopping initialization
    n_epochs_stop_acc = 5
    epochs_no_improve_acc = 0
    max_val_acc = 0

    for epoch in range(epochs):
    # training
        for batch_idx, (x, target) in enumerate(train_dataloader):
            model.train()

            optimizer.zero_grad()
            x, target = Variable(x), Variable(target)
            x, target = x.to(device), target.to(device)
            out = model(x)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                val_loss, accuracy = evaluate_model(model, device, val_dataloader, optimizer, criterion)
                print('epoch {} batch {}  [{}/{}] training loss: {:1.4f} \tvalidation loss: {:1.4f}\tAccuracy (val): {:.1%}'.format(epoch,batch_idx,batch_idx*len(x),
                        len(train_dataloader.dataset),loss.item(), val_loss, accuracy))

        # Get the new validation accuracy
        val_loss, accuracy = evaluate_model(model, device, val_dataloader, optimizer, criterion)
        print("Accuracy (val): {:.1%}".format(accuracy))

        # Early stopping accuracy tracking        
        if accuracy > max_val_acc:
            epochs_no_improve_acc = 0
            max_val_acc = accuracy
        else:
            epochs_no_improve_acc += 1
        
        if epoch >= 10 and epochs_no_improve_acc >= n_epochs_stop_acc:
            print('Early stopping !')
            print("Stopped")
            break
        else:
            continue
    
    # Get the last validation accuracy
    val_loss, accuracy = evaluate_model(model, device, val_dataloader, optimizer, criterion)

    return accuracy

