import numpy as np
import torch
from torch.autograd import Variable
#from ray import tune

#metrics dice coefficient
def compute_dice(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return intersection / union

def evaluate_model(model, device,val_loader, optimizer, criterion):
    model.eval()
    num_batches = 0
    avg_loss = 0
    dice = 0
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            data = sample['image']
            target = sample['mask']
            data, target = data.to(device),target.to(device)
            output = model(data)
            # Dice computation
            out_cut = np.copy(output.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            val_dice = compute_dice(out_cut, target.data.cpu().numpy())
            dice += val_dice
            ## LOSS
            loss = criterion(output, target)
            avg_loss += loss.item()
            num_batches += 1
    avg_loss /= num_batches
    dice /= num_batches
    return avg_loss, dice


def train_segmentation(model, device, train_loader,val_loader, optimizer, criterion, epochs=20):
    # Training loss and dice lists
    loss_history = []
    dice_train_history = []

    # Validation loss and dice lists for each epochs
    val_dice_history = []
    val_loss_history = []
    for epoch in range(epochs):
        num_batches = 0
        train_dice = []
        losses=[]
        for idx, sample in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            data = sample['image']
            target = sample['mask']
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Dice computation
            out_cut = np.copy(output.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            batch_train_dice = compute_dice(out_cut, target.data.cpu().numpy())
            train_dice.append(batch_train_dice)
            ## LOSS
            loss = criterion(output, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if idx % 20 == 0:
                val_loss, val_dice = evaluate_model(model, device, val_loader, optimizer, criterion)
                print('epoch {} batch {}  [{}/{}]\ttraining loss: {:1.4f} \tvalidation loss: {:1.4f}\t\tDice (val): {:.1%}'.format(epoch, idx, idx*len(data),
                        len(train_loader.dataset), loss.item(), val_loss, val_dice))
        
        loss_history.append(np.array(losses).mean())
        dice_train_history.append(np.array(train_dice).mean())

        # Get the new validation accuracy
        val_loss, val_dice = evaluate_model(model, device, val_loader, optimizer, criterion)
        val_dice_history.append(val_dice * 100)
        val_loss_history.append(val_loss)
        print("Loss (val): {:1.4f}".format(val_loss))
        print("Dice (val): {:.1%}".format(val_dice))
        
    return val_loss_history, val_dice_history
