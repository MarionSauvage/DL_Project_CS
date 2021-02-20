import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import f1_score
from classification import *


def evaluate_model(model, device, dataloader, optimizer, criterion):
    avg_accuracy = 0
    avg_loss = 0.0

    # numpy arrays to compute the f1-score
    preds_array = np.empty(0, dtype=np.int64)
    targets_array = np.empty(0, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)

            loss = criterion(out, targets)
            _, predictions = torch.max(out, 1)
            nb_correct = torch.sum(predictions == targets)

            preds_array = np.concatenate((preds_array, predictions.cpu().detach().numpy()))
            targets_array = np.concatenate((targets_array, targets.cpu().detach().numpy()))

            avg_loss += loss.item()
            avg_accuracy += nb_correct
    
    return avg_loss / len(dataloader.dataset), float(avg_accuracy) / len(dataloader.dataset), f1_score(targets_array, preds_array)


def train_classification(model, device, train_dataloader, val_dataloader, optimizer, criterion, epochs=20):
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
                val_loss, val_acc, val_f1_score = evaluate_model(model, device, val_dataloader, optimizer, criterion)
                print('epoch {} batch {}  [{}/{}] training loss: {:1.4f} \tvalidation loss: {:1.4f}\tAccuracy (val): {:.1%}\tF1-score (val): {:.1%}'.format(epoch,batch_idx,batch_idx*len(x),
                        len(train_dataloader.dataset),loss.item(), val_loss, val_acc, val_f1_score))
    
    # Get the last validation accuracy
    val_loss, val_acc, val_f1_score = evaluate_model(model, device, val_dataloader, optimizer, criterion)

    return val_acc, val_f1_score

