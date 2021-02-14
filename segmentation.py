import numpy as np
import torch
from torch.autograd import Variable
#from ray import tune

#metrics IOU/Jaccard Index
def compute_iou(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return intersection / union

def evaluate_model(model, device,val_loader, optimizer, criterion):
    model.eval()
    num_batches = 0
    avg_loss = 0
    iou=0
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            data = sample['image']
            target = sample['mask']
            data, target = data.to(device),target.to(device)
            output = model(data)
            #IOU computation
            out_cut = np.copy(output.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            val_iou = compute_iou(out_cut, target.data.cpu().numpy())
            iou+=val_iou
            ## LOSS
            loss = criterion(output, target)
            avg_loss += loss.item()
            num_batches += 1
    avg_loss /= num_batches
    iou /= num_batches
    return avg_loss, iou


def train_segmentation(model, device, train_loader,val_loader, optimizer, criterion, epochs=20):
    loss_history = []
    val_iou_history = []
    iou_train_history = []
    for epoch in range(epochs):
        num_batches = 0
        train_iou = []
        losses=[]
        for idx, sample in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            data = sample['image']
            target = sample['mask']
            data, target = data.to(device), target.to(device)
            output = model(data)
            #IOU computation
            out_cut = np.copy(output.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            train_dice = compute_iou(out_cut, target.data.cpu().numpy())
            train_iou.append(train_dice)
            ## LOSS
            loss = criterion(output, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if idx % 20 == 0:
                val_loss, val_iou = evaluate_model(model, device, val_loader, optimizer, criterion)
                val_iou_history.append(val_iou)

                print('epoch {} batch {}  [{}/{}]\ttraining loss: {:1.4f} \tvalidation loss: {:1.4f}\t\tIoU (val): {:.1%}'.format(epoch, idx, idx*len(data),
                        len(train_loader.dataset), loss.item(), val_loss, val_iou))
        
        loss_history.append(np.array(losses).mean())
        iou_train_history.append(np.array(train_iou).mean())

        # Get the new validation accuracy
        val_loss, val_iou = evaluate_model(model, device, val_loader, optimizer, criterion)
        val_iou_history.append(val_iou)
        print("Loss (val): {:1.4f}".format(val_loss))
        print("IoU (val): {:.1%}".format(val_iou))
        
    return loss_history, iou_train_history, val_iou_history
