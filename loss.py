#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yiming Yang
# Date: 08/07/2022
# Email: y.yang2@napier.ac.uk
# Description: define various loss functions and evaluation metrics 

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


smooth = 1e-6 # avoid 0 division
alpha = 0.8 # parameters in Focal loss
beta = 0.5 # parameters in Tversky loss
gamma = 1. # parameters in Tversky loss


    
def pixel_accuracy(output, mask):
    """ 
    Pixelwise Accuracy 

    Args:
        output (Tensor with dims (b, c, h, w)): predicated mask
        mask (Tensor with dims (b, c, h, w)): ground truth mask 

    Returns:
        pixelwise accuracy (float)
    """
    with torch.no_grad():
        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=6):
    """
    mean Intersection over Union (mIoU) 
    remark: if multi-class problem return the mean over each class

    Args:
        pred_mask (Tensor with dims (b, c, h, w)): predicated mask
        mask (Tensor with dims (b, c, h, w)): ground truth mask
        smooth (_type_, optional): avoid zero division. Defaults to 1e-10.
        n_classes (int, optional): number of object classes. Defaults to 6.

    Returns:
        mIoU (float)
    """

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
    # CrossEntropy Loss augmented by a map with pixel-wise loss weights to help seperate overlapped cells 
    # Implementation and Mathematical Details:
    # CrossEntropy Loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # Loss Map: https://arxiv.org/pdf/1505.04597.pdf

    def __init__(self, reduction='mean', class_weight=None):
        # reduction (string, optional): reduction to apply to the output: {'none' | 'mean' | 'sum'}. 
        # - default is 'mean'
        # class_weight (tensor, optional): define the loss weights on each class. Lenght should be same as number of object classes 
        # - default is None
        super(Weighted_Pixel_Wise_CrossEntropyLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction
        self.class_weight = class_weight

    def forward(self, preds, targets, pixelwise_weights=None):
        # preds (Tensor with dims (b, c, h, w)): predicated mask
        # targets (Tensor with dims (b, h, w)): ground truth mask
        # pixelwise_weights (Tensor with dims (b, h, w)): pixelwise loss weights
        
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

        # Average over mini-batch
        weighted_loss = -1.0*weighted_logp.mean()
        
        return weighted_loss


class Weighted_Pixel_Wise_FocalLoss(nn.Module):
    # Focal Loss augmented by a map with pixel-wise loss weights to help seperate overlapped cells 
    # Implementation and Mathematical Details:
    # Focal Loss: https://arxiv.org/abs/1708.02002v2
    # Loss Map: https://arxiv.org/pdf/1505.04597.pdf
    def __init__(self, reduction='mean', class_weight=None):
        # reduction (string, optional): reduction to apply to the output: {'none' | 'mean' | 'sum'}. 
        # - default is 'mean'
        # class_weight (tensor, optional): define the loss weights on each class. Lenght should be same as number of object classes 
        # - default is None
        super(Weighted_Pixel_Wise_FocalLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction
        self.class_weight = class_weight

    def forward(self, preds, targets, pixelwise_weights=None, alpha=alpha, gamma=gamma):
        # preds (Tensor with dims (b, c, h, w)): predicated mask
        # targets (Tensor with dims (b, h, w)): ground truth mask
        # pixelwise_weights (Tensor with dims (b, h, w)): pixelwise loss weights
        # alpha (int, ): focal loss parameter - default: 0.8
        # gamma (int, ): focal loss parameter - default: 1.0

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
        






