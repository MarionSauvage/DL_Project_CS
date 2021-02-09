import torch
from torch.autograd import Variable
from ray import tune
from preprocessing_for_classification import *

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
    for idx, sample in enumerate(val_loader):
        data = sample['image']
        target = sample['mask']
        data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
        output = model.forward(data)
        # output = (output > 0.5).type(opt.dtype)	# use more gpu memory, also, loss does not change if use this line
        loss = criterion(output, target)
        avg_loss += loss.data[0]
        num_batches += 1
    avg_loss /= num_batches
    print('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss))


def train_segmentation(model, device, train_loader,val_loader, optimzer, criterion, epochs=20):
    for epoch in range(epochs):
        model.train()
        num_batches = 0
        avg_loss = 0
        for idx, sample in enumerate(train_loader):
            data = sample['image']
            target = sample['mask']
            data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
            optimizer.zero_grad()
            output = model(data)
            # output = (output > 0.5).type(opt.dtype)	# use more gpu memory, also, loss does not change if use this line
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.data[0]
            if batch_idx % 20 == 0:
                    val_loss, accuracy = evaluate_model(model, device, val_dataloader, optimizer, criterion)
                    print('epoch {} batch {}  [{}/{}] training loss: {:1.4f} \tvalidation loss: {:1.4f}\tAccuracy (val): {:.1%}'.format(epoch,batch_idx,batch_idx*len(x),
                            len(train_dataloader.dataset),loss.item(), val_loss, accuracy))


# make prediction
def run_test(model, test_loader, opt):
    """
    predict the masks on testing set
    :param model: trained model
    :param test_loader: testing set
    :param opt: configurations
    :return:
        - predictions: list, for each elements, numpy array (Width, Height)
        - img_ids: list, for each elements, an image id string
    """
    model.eval()
    predictions = []
    img_ids = []
    for batch_idx, sample_batched in enumerate(test_loader):
        data, img_id, height, width = sample_batched['image'], sample_batched['img_id'], sample_batched['height'], sample_batched['width']
        data = Variable(data.type(opt.dtype))
        output = model.forward(data)
        # output = (output > 0.5)
        output = output.data.cpu().numpy()
        output = output.transpose((0, 2, 3, 1))    # transpose to (B,H,W,C)
        for i in range(0,output.shape[0]):
            pred_mask = np.squeeze(output[i])
            id = img_id[i]
            h = height[i]
            w = width[i]
            # in p219 the w and h above is int
            # in local the w and h above is LongTensor
            if not isinstance(h, int):
                h = h.cpu().numpy()
                w = w.cpu().numpy()
            pred_mask = resize(pred_mask, (h, w), mode='constant')
            pred_mask = (pred_mask > 0.5)
            predictions.append(pred_mask)
            img_ids.append(id)

    return predictions, img_ids
