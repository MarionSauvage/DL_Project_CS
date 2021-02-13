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
    for idx, sample in enumerate(val_loader):
        data = sample['image']
        target = sample['mask']
        data, target = data.to(device),target.to(device)
        output = model.forward(data)
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
    n_epochs_stop = 6
    epochs_no_improve = 0
    early_stop = False
    min_val_loss = np.Inf
    loss_history = []
    track_loss=0
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
            #Early stopping loss tracking
            track_loss+=loss
            track_loss=track_loss/len(train_loader)
            if track_loss < min_val_loss:
                epochs_no_improve=0
                min_val_loss=track_loss
            else:
                epochs_no_improve+=1
            if epoch>10 and epochs_no_improve==n_epochs_stop:
                print('Early stopping !')
                early_stop=True
                break
            else:
                continue
            loss.backward()
            optimizer.step()
            if idx % 20 == 0:
                val_loss, val_iou = evaluate_model(model, device, val_loader, optimizer, criterion)
                val_iou_history.append(val_iou)

                print('epoch {} batch {}  [{}/{}]\ttraining loss: {:1.4f} \tvalidation loss: {:1.4f}\t\tIoU (val): {:.1%}'.format(epoch, idx, idx*len(data),
                        len(train_loader.dataset), loss.item(), val_loss, val_iou))
        
        loss_history.append(np.array(losses).mean())
        iou_train_history.append(np.array(train_iou).mean())
        

        # Get the last validation accuracy
        val_loss, val_iou = evaluate_model(model, device, val_loader, optimizer, criterion)
        val_iou_history.append(val_iou)
        if early_stop:
            print("Stopped")
            break
    return loss_history, iou_train_history, val_iou_history
