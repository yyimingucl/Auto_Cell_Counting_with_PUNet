#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yiming Yang
# Date: 08/07/2022
# Email: y.yang2@napier.ac.uk
# Description: meta parameters 



class hyper_param:
    image_size = (256, 256)
    prob_adjust_brightness = 0.2
    prob_adjust_contrast = 0.2
    prob_adjust_gamma = 0.2
    prob_rotate_90 = 0.3
    prob_rotate_180 = 0.3
    prob_rotate_270 = 0.3
    prob_hflip = 0.3
    prob_vflip = 0.3

    # Training Parameter
    num_epochs = 40 # 30 for Fluorescent and Kaggle (UNet and Res_UNet) 50 for CoNIC // 40 for Fluorescent and Kaggle (P_UNet) 60 for CoNIC
    lr = 1e-4
    batch_size = 16 # 64 for UNet_v2 and Res_UNet // 16 for P_UNet
    num_folds = 10
