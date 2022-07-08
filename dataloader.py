#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yiming Yang
# Date: 08/07/2022
# Email: y.yang2@napier.ac.uk
# Description: define dataloader for three different datasets {'Kaggle Data Science 2018'|'Fluorscent Dataset'|'CoNIC'}

import os
import torch
import numpy as np
from random import random
from parameter import hyper_param
from torch.utils.data import Dataset
from torch.nn import ModuleList
from torchvision.transforms  import (RandomApply, RandomRotation, Compose, RandomHorizontalFlip, 
                                     RandomVerticalFlip, Normalize, RandomCrop)
from torchvision.transforms.functional import (adjust_brightness, adjust_gamma, adjust_contrast)


class DATASET_BASE(Dataset):
    # Base Dataloader Class
    def __init__(self, if_crop=False):
        self.if_crop = if_crop
        self.height, self.width = hyper_param.image_size
        self.prob_adjust_brightness = hyper_param.prob_adjust_brightness
        self.prob_adjust_gamma = hyper_param.prob_adjust_gamma
        self.prob_adjust_contrast = hyper_param.prob_adjust_contrast
        self.prob_rotate_90 = hyper_param.prob_rotate_90
        self.prob_rotate_180 = hyper_param.prob_rotate_180
        self.prob_rotate_270 = hyper_param.prob_rotate_270
        self.prob_hflip = hyper_param.prob_hflip
        self.prob_vflip = hyper_param.prob_vflip
        self.transformer = self.create_transformer()


    def create_transformer(self):
        # define transformation for input image 
        Rotate_90 = RandomApply(ModuleList([RandomRotation((90,90)),]),p=self.prob_rotate_90)
        Rotate_180 = RandomApply(ModuleList([RandomRotation((180,180)),]),p=self.prob_rotate_180)
        Rotate_270 = RandomApply(ModuleList([RandomRotation((270,270)),]),p=self.prob_rotate_270)
        if self.if_crop:
            return Compose([RandomCrop((self.height, self.width)),#config.input_size),
                            RandomHorizontalFlip(p=self.prob_hflip),
                            RandomVerticalFlip(p=self.prob_vflip),
                            Rotate_90,
                            Rotate_180,
                            Rotate_270])
        else:
            return Compose([RandomHorizontalFlip(p=self.prob_hflip),
                            RandomVerticalFlip(p=self.prob_vflip),
                            Rotate_90,
                            Rotate_180,
                            Rotate_270])


    def gen_data(self, image, mask, loss_weight=None):    
        # generate data 
        image = image.permute(2,0,1)
        mask = mask.unsqueeze(0)


        if random() < self.prob_adjust_brightness:
            image = adjust_brightness(image, np.random.uniform(0.5,1.5))
        if random() < self.prob_adjust_gamma:
            image = adjust_gamma(image, gamma=np.random.uniform(0.5,1.5))
        if random() < self.prob_adjust_contrast:
            image = adjust_contrast(image, np.random.uniform(0.5,1.5))


        if loss_weight is not None:
            data =  torch.cat((image, mask, loss_weight.unsqueeze(0)), dim=0)
            data = self.transformer(data)
            data = torch.split(data, [3, 1, 1], dim=0)
            image, mask, loss_weight = data[0], data[1], data[2]

            mask = mask.squeeze(0)
            loss_weight = loss_weight.squeeze(0)

            return image, mask.type(torch.long), loss_weight
        else:
            data = torch.cat((image, mask), dim=0)
            data = self.transformer(data)
            data = torch.split(data, [3, 1], dim=0)

            mask = mask.squeeze(0)

            return image, mask.type(torch.long)
    

# Dataset for CoNIC Challenge
class CoNIC_DATASET(DATASET_BASE):
    def __init__(self, images, masks, counts, num_class=6):
        super(CoNIC_DATASET, self).__init__(if_crop=False)
        self.images = torch.tensor(images) / 255.
        self.masks = torch.tensor(masks)
        self.loss_weights = None
        self.num_class = num_class
        self.counts = counts

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        mask = self.masks[idx,:,:]
        count = self.counts[idx]

        if self.loss_weights is None:
            image, mask = self.gen_data(image, mask)
            return image, mask, count
        else:
            loss_weight = self.loss_weights[idx,:,:]
            image, mask, loss_weight = self.gen_data(image, mask, loss_weight)
            return image, mask, loss_weight, count

    def obtain_loss_weights(self, loss_weights):
        self.loss_weights = torch.tensor(loss_weights)
        assert len(self.loss_weights) == self.__len__(), "The number of loss weights {} is not consistent with number of samples {}".format(len(self.loss_weights), self.__len__())

    
        

# Dataset for flour
class Kaggle_DATASET(DATASET_BASE):
    def __init__(self, base_path, sample_sets, counts, if_use_loss_weight=False):
        super(Kaggle_DATASET, self).__init__(if_crop=False)
        self.base_path = base_path
        self.sample_sets = sample_sets
        self.counts = counts
        self.if_use_loss_weight = if_use_loss_weight


    def __len__(self):
        return len(self.sample_sets)
    
    def __getitem__(self, idx):
        self.idx = idx
        sample_path = os.path.join(self.base_path, self.sample_sets[idx])
        count = self.counts[self.sample_sets[idx]]

        if self.if_use_loss_weight:
            image, mask, loss_weight = self.obtain_data(sample_path)
            image, mask, loss_weight = self.gen_data(image, mask, loss_weight)
            return image, mask, loss_weight, count
        else:
            image, mask = self.obtain_data(sample_path)
            image, mask = self.gen_data(image, mask)
            return image, mask, count


    def obtain_data(self, sample_path):
        # Load Mask
        mask_folder = os.path.join(sample_path, 'masks')
        mask_path = os.listdir(mask_folder)[0]
        mask_path = os.path.join(mask_folder, mask_path)
        mask = np.load(mask_path).astype(np.int32)
        
        # Load Image
        image_folder = os.path.join(sample_path, 'images')
        image_path = os.listdir(image_folder)[0]
        image_path = os.path.join(image_folder, image_path)
        image = np.load(image_path).astype(np.float32)
        image = image / 255.

        # Load Loss Weights if required 
        if self.if_use_loss_weight:
            loss_weight_folder = os.path.join(sample_path, 'loss_weights')
            loss_weight_path = os.listdir(loss_weight_folder)[0]
            loss_weight_path = os.path.join(loss_weight_folder, loss_weight_path)
            loss_weight = np.load(loss_weight_path).astype(np.float32)
            return torch.tensor(image), torch.tensor(mask), torch.tensor(loss_weight)
        
        else:
            return torch.tensor(image), torch.tensor(mask)

# Dataset for Flourscent Data        
class Flourscent_DATASET(DATASET_BASE):
    def __init__(self, base_path, sample_sets, counts, if_use_loss_weight=False):
        super(Flourscent_DATASET, self).__init__(if_crop=False)
        self.base_path = base_path
        self.sample_sets = sample_sets
        self.counts = counts
        self.if_use_loss_weight = if_use_loss_weight
    
    def __len__(self):
        return len(self.sample_sets)
    
    def __getitem__(self, idx):
        sample_id = self.sample_sets[idx]

        sample_image_path = os.path.join(self.base_path+'/all_images/images', sample_id)
        sample_mask_path = os.path.join(self.base_path+'/all_masks/masks', sample_id)
        count = self.counts[sample_id]

        image = np.load(sample_image_path).astype(np.float32) / 255.
        image = torch.tensor(image)

        mask = np.load(sample_mask_path).astype(np.int32)
        mask = torch.tensor(mask)

        if self.if_use_loss_weight:
            sample_loss_weight_path = os.path.join(self.base_path+'/all_loss_weights/loss_weights', sample_id) 
            loss_weight = np.load(sample_loss_weight_path).astype(np.float32)
            loss_weight = torch.tensor(loss_weight)

            image, mask, loss_weight = self.gen_data(image, mask, loss_weight)
            return image, mask, loss_weight, count
        
        else:
            image, mask = self.gen_data(image, mask)
            return image, mask, count


        

        
