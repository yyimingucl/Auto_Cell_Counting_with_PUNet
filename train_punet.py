#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yiming Yang
# Date: 08/07/2022
# Email: y.yang2@napier.ac.uk
# Description: Training script for Probabilistic UNet on Kaggle Dataset, Fluorscent Dataset, and CoNIC Dataset.

import os
import torch 
import json
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from Model import l2_regularisation
from torch.utils.tensorboard import SummaryWriter
from parameter import hyper_param


model_name = 'Probabilistic_UNet' 
dataset_name = 'Flourscent'  # Kaggle Flourscent CoNIC

config = {'model': model_name,
          'num_epochs':hyper_param.num_epochs, 
          'loss':'weighted_ce_loss', 
          'lr':hyper_param.lr,
          'model_save_folder':'scratch/model_weight/{}/{}'.format(dataset_name, model_name),
          'log_save_folder':'scratch/train_log/{}/{}'.format(dataset_name, model_name),
          'dataset': dataset_name,
          'use_loss_weight': True,
          'train/valid_split_rate': 0.1, 
          'batch_size': hyper_param.batch_size, 
          'optimizer': 'Adam'}


# Set up Paths for storing results
if not os.path.exists(config['model_save_folder']):
    os.mkdir(config['model_save_folder'])
if not os.path.exists(config['log_save_folder']):
    os.mkdir(config['log_save_folder'])



print('[INFO] Model Name: {}'.format(model_name))
#Set up Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('[INFO] Loading Device: {}'.format(device))


# Set up Dataset
if config['dataset'] == 'CoNIC':
    from dataloader import CoNIC_DATASET

    num_class = 6
    print('[INFO] Building DataLoader for CoNIC Dataset')

    image_file = '/home/y_yang/scratch/CoNIC_Data/conic_images.npy'
    images = np.load(image_file).astype(np.float32)

    mask_file = '/home/y_yang/scratch/CoNIC_Data/conic_masks.npy'
    masks = np.load(mask_file)[:,:,:,1].astype(np.int8)

    count_file = '/home/y_yang/scratch/CoNIC_Data/conic_counts.csv'
    counts = np.genfromtxt(count_file, delimiter=',', skip_header = 1).astype(np.int8)

    dataset = CoNIC_DATASET(images, masks, counts=counts, num_class=num_class)

    if config['use_loss_weight']:
        loss_weight_file = '/home/y_yang/scratch/CoNIC_Data/conic_loss_weights.npy'
        loss_weight = np.load(loss_weight_file).astype(np.float32)
        dataset.obtain_loss_weights(loss_weight)
    else:
        loss_weight = None

elif config['dataset'] == 'Kaggle':
    from dataloader import Kaggle_DATASET

    num_class = 1
    print('[INFO] Building DataLoader for 2018 Data Science Bowl Dataset')
    
    base_path = '/home/y_yang/scratch/2018_Data_Science_Bowl/stage1_train'
    sample_sets = sorted(os.listdir(base_path))
    counts = json.load(open('/home/y_yang/scratch/2018_Data_Science_Bowl/kaggle_counts.txt'))

    dataset = Kaggle_DATASET(base_path, sample_sets, counts, config['use_loss_weight'])

elif config['dataset'] == 'Flourscent':
    from dataloader import Flourscent_DATASET

    num_class = 1
    print('[INFO] Building DataLoader for Flourscent Microscopy Dataset')

    base_path = '/home/y_yang/scratch/Flourscent_Data'
    sample_sets = sorted(os.listdir('/home/y_yang/scratch/Flourscent_Data/all_images/images'))

    counts_path = os.path.join(base_path, 'flourscent_counts.txt')
    counts = json.load(open(counts_path))

    dataset = Flourscent_DATASET(base_path, sample_sets, counts=counts, if_use_loss_weight=config['use_loss_weight'])

else:
    raise NameError('[WARNING] Not Known DataSet Name') 


# Set up Loss
if config['loss'] == 'weighted_ce_loss':
    print('[INFO] Loss Function: Weighted Cross Entropy Loss')
    from loss import Weighted_Pixel_Wise_CrossEntropyLoss
    criterion = Weighted_Pixel_Wise_CrossEntropyLoss()
elif config['loss'] == 'weighted_dice_ce_loss':
    print('[INFO] Loss Function: Weighted Dice + Cross Entropy Loss')
    from loss import Weighted_Pixel_Wise_DiceCELoss
    criterion = Weighted_Pixel_Wise_DiceCELoss()
elif config['loss'] == 'weighted_focal_loss':
    print('[INFO] Loss Function: Weighted Focal Loss')
    from loss import Weighted_Pixel_Wise_FocalLoss
    criterion = Weighted_Pixel_Wise_FocalLoss()
else:
    raise NameError('[Warning] Not Known Loss Function')


# Set up K-Folder 
dataset_size = len(dataset)
indices = list(range(dataset_size))
kfold = KFold(n_splits=hyper_param.num_folds, shuffle=True)



print('[INFO] Start Training')
for fold, (train_ids, valid_ids) in enumerate(kfold.split(indices)):

    print(f'[INFO] FOLD {fold}')
    print(' ')

    # Set up Model
    from Model.Probabilistic_UNet import ProbabilisticUnet
    model = ProbabilisticUnet(num_in_channels=3, num_classes=num_class+1, latent_dim=2*(num_class+1), criterion=criterion)
    model.to(device)

    # Set up optimizer
    if config['optimizer'] == 'Adam':
        # print('[INFO] Optimizer: Adam, Learning Rate: {}'.format(config['lr']))
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=config['lr'])

    elif config['optimizer'] == 'SGD':
        # print('[INFO] Optimizer: SGD, Learning Rate: {}'.format(config['lr']))
        from torch.optim import SGD
        optimizer = SGD(model.parameters(), lr=config['lr'])  

    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)

    
    train_sampler = SubsetRandomSampler(train_ids)
    valid_sampler = SubsetRandomSampler(valid_ids)
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=valid_sampler)


    # Save_Path
    run_id = '{}_{}'.format(config['model'], fold)

    log_save_path = os.path.join(config['log_save_folder'], run_id) 

    model_name = '{}_{}_{}_{}.pt'.format(config['model'], config['loss'], config['dataset'], fold)
    model_save_path = os.path.join(config['model_save_folder'], model_name)

    indices_path = '/home/y_yang/scratch/training_ids/{}/{}'.format(config['dataset'], config['model'])
    if not os.path.exists(indices_path):
        os.mkdir(indices_path)
    indices_path = os.path.join(indices_path, run_id+'_{}'.format(config['dataset']))
    if not os.path.exists(indices_path):
        os.mkdir(indices_path)

    log_indices = {'training':train_ids.tolist(), 'valid':valid_ids.tolist()}
    json.dump(log_indices, open(os.path.join(indices_path, "train_valid_ids.txt"),'w'))  


    #Set up tensorboard
    writer = SummaryWriter(log_save_path)

    for epoch in range(config['num_epochs']): 
        print(' ')
        print('[INFO] {}th Epoch Start'.format(epoch+1))

        # -----------------------------  Training ------------------------------------ #
        model.train()
        train_iou_score = 0
        train_accuracy = 0
        train_loss = 0
        
        for package in train_loader:

            data = package[0].to(device)
            target = package[1].to(device)
            target = target.unsqueeze(1)
            count = package[-1]

            if len(package) == 3:
                loss_weight = None
            elif len(package) == 4:
                loss_weight = package[2].to(device)
            
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            model.forward(data, target, training=True)
            elbo = model.elbo(target, loss_weight=loss_weight)
            reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss

            loss.backward()

            optimizer.step()
            
            train_loss += loss.item()

        # -------------------------------- Logging Stats ------------------------------ #
        train_loss = train_loss / train_loader.__len__()
        train_iou_score = train_iou_score / train_loader.__len__()
        train_accuracy = train_accuracy / train_loader.__len__()


        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('mIoU/Train', train_iou_score, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
 
        print('[INFO] Train loss: {:.5f}.'.format(train_loss))

        torch.save(model.state_dict(), model_save_path)    
        print(' ')
