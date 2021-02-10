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
        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
        val_iou = compute_iou(out_cut, target.data.cpu().numpy())
        iou+=val_iou
        ## LOSS
        loss = criterion(output, target)
        avg_loss += loss.data[0]
        num_batches += 1
    avg_loss /= num_batches
    print('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss))
    return val_iou/idx,avg_loss


def train_segmentation(model, device, train_loader,val_loader, optimizer, criterion, epochs=20):
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(epochs):
        num_batches = 0
        avg_loss = 0
        train_iou = []
        losses=[]
        for idx, sample in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            data = sample['image']
            target = sample['mask']
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(output.shape)
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
            avg_loss += loss.data[0]
            if idx % 20 == 0:
                    val_loss= evaluate_model(model, device, val_loader, optimizer, criterion)

                    print('epoch {} batch {}  [{}/{}] training loss: {:1.4f} \tvalidation loss: {:1.4f:}'.format(epoch,idx,idx*len(x),
                            len(train_loader.dataset),loss.item(), val_loss))
        
        loss_history.append(np.array(losses).mean())
        iou_train_history.append(np.array(train_iou).mean())
        val_loss_history.append(val_mean_iou)
    return loss_history, iou_train_history,val_loss_history
