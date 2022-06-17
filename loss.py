import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


smooth = 1e-6 # avoid 0 division
alpha = 0.8 # parameters in Focal loss
beta = 0.5 # parameters in Tversky loss
gamma = 1. # parameters in Tversky loss


def dice_coef_multiclass(preds, targets, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    # preds = preds[:,1:,:,:]
    preds = preds[:,1:,:,:]
    preds = torch.sigmoid(preds.permute((0,2,3,1)))
    # print(preds.shape)
    preds = torch.flatten(preds)
    #preds = preds.view(-1)
    # targets = targets[:,1:,:,:]
    targets = nn.functional.one_hot(targets, num_classes = -1)
    targets = torch.flatten(targets[:,:,:,1:])
    # targets = targets[:,:,:,1:].view(-1)

    # intersect = torch.sum(preds * targets, axis=-1)
    # denom = torch.sum(preds + targets, axis=-1)  

    intersect = (preds*targets).sum()
    return torch.mean(( (2. * intersect + smooth) / (preds.sum() + targets.sum() + smooth)))

def dice_coef_multiclass_loss(preds, targets):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_multiclass(preds, targets)



def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=6):
    with torch.no_grad():
        pred_mask = torch.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes+1): #loop per pixel class
            true_class = pred_mask == clas 
            true_label = mask == clas 

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)






class Weighted_Pixel_Wise_CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', class_weight=None):
        super(Weighted_Pixel_Wise_CrossEntropyLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction
        self.class_weight = class_weight

    def forward(self, preds, targets, pixelwise_weights=None):
        if len(targets.shape) == 3:   
            batch_size, H, W = targets.shape
        elif len(targets.shape) == 4:
            batch_size, _, H, W = targets.shape

        # Calculate log probabilities
        # logp = self.log_softmax(preds)
        logp = F.log_softmax(preds, dim=1)

        # Gather log probabilities with respect to target
        logp = logp.gather(1, targets.view(batch_size, 1, H, W))


        if pixelwise_weights != None: 
            pixelwise_weights = Variable(pixelwise_weights)
            # Multiply with weights
            weighted_logp = (logp * pixelwise_weights).view(batch_size, -1)

        elif pixelwise_weights == None:
            weighted_logp = (logp).view(batch_size, -1)


        # Average over mini-batch
        weighted_loss = -1.0*weighted_logp.mean()
        
        return weighted_loss


class Weighted_Pixel_Wise_FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Weighted_Pixel_Wise_FocalLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, preds, targets, pixelwise_weights=None, alpha=alpha, gamma=gamma):

        if len(targets.shape) == 3:   
            batch_size, H, W = targets.shape
        elif len(targets.shape) == 4:
            batch_size, _, H, W = targets.shape

        # Calculate log probabilities
        logp = F.log_softmax(preds, dim=1)

        # Gather log probabilities with respect to target
        logp = logp.gather(1, targets.view(batch_size, 1, H, W))

        if pixelwise_weights != None: 
            pixelwise_weights = Variable(pixelwise_weights)
            # Multiply with weights
            weighted_logp = (logp * pixelwise_weights).view(batch_size, -1)

        elif pixelwise_weights == None:
            weighted_logp = (logp).view(batch_size, -1)

        CE_weighted_loss = -1.0*weighted_logp.mean()

        CE_EXP = torch.exp(-CE_weighted_loss)
        focal_loss = alpha * (1-CE_EXP)**gamma * CE_weighted_loss

        return focal_loss
        


class Weighted_Pixel_Wise_DiceCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Weighted_Pixel_Wise_DiceCELoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, preds, targets, pixelwise_weights=None):
        if len(targets.shape) == 3:   
            batch_size, H, W = targets.shape
        elif len(targets.shape) == 4:
            batch_size, _, H, W = targets.shape

        # Calculate log probabilities
        logp = F.log_softmax(preds, dim=1)

        # Gather log probabilities with respect to target
        logp = logp.gather(1, targets.view(batch_size, 1, H, W))

        if pixelwise_weights != None: 
            pixelwise_weights = Variable(pixelwise_weights)
            # Multiply with weights
            weighted_logp = (logp * pixelwise_weights).view(batch_size, -1)

        elif pixelwise_weights == None:
            weighted_logp = (logp).view(batch_size, -1)


        # weighted_logp = weighted_logp.sum(1) / pixelwise_weights.view(batch_size, -1).sum(1)

        # Average over mini-batch
        weighted_loss = -1.0*weighted_logp.mean()
        
        # preds = torch.softmax(preds, dim=1)# Pytorch assume the 0th dim as batch size.     
        dice_loss = dice_coef_multiclass_loss(preds, targets)
        
        
        Dice_CE = weighted_loss + dice_loss
        
        return Dice_CE