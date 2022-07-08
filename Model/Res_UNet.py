#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yiming Yang
# Date: 08/07/2022
# Email: y.yang2@napier.ac.uk
# Description: Define the Res_UNet Model and Conditional Random Field (CRF) as post processing.
#              1. Res_UNet Referenced from Roberto Morelli et al. (2021): Automating cell counting in fuorescent microscopy
#              through deep learning with c‑ResUnet (https://www.nature.com/articles/s41598-021-01929-5.pdf)
#              2. CRF Referenced from Shuai Zheng et al. (2014): Conditional Random Fields as Recurrent Neural Networks
#              (https://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf)
#                 CRF Implementation: https://github.com/HapeMask/crfrnn_layer 




import torch
import numpy as np
from torch.nn import (Module,Conv2d, MaxPool2d, Conv2d, AvgPool2d, ConvTranspose2d)
from .Blocks import _Conv_BN_Activation, _Resnet_Conv_BN_ReLU, init_weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##########################################################################################################################
##############################                            RES_UNet                        ################################
##########################################################################################################################

class Res_UNet(Module):
    def __init__(self, num_in_channels=3, num_out_channels=7, activation='elu', pooling = 'max', apply_final_layer=False):
        """
        Res_UNet Model

        Args:
            num_in_channels (int, optional): number of input channels. Defaults to 3.
            
            num_out_channels (int, optional): number of output channels. Defaults to 7 (number of object classes + 1).
            
            activation (str, optional): activation function. Defaults to 'elu'. {'elu', 'relu', 'identity'}
            
            pooling (str, optional): pooling layer. Defaults to 'max'. {'max','average'}
            
            apply_final_layer (bool, optional): whether use the final layer. If False, the number of output channels 
            will be 32 and post-processed by CRF, otherwise return the mask with number of channels equal to class number + 1
            Defaults to False.
        """
        super(Res_UNet, self).__init__()

        if pooling == 'max':
            self.pool = MaxPool2d((2, 2), stride=2)
        elif pooling == 'average':
            self.pool = AvgPool2d((2, 2), stride=2)

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.apply_final_layer = apply_final_layer
        

        self.conv_block = _Conv_BN_Activation(num_in_channels=self.num_in_channels, num_out_channels=32, kernel_size=(5, 5), activation=activation)       
        self.resnet_block_1 = _Resnet_Conv_BN_ReLU(num_in_channels=32, num_out_channels=64, kernel_size=(5, 5), activation=activation)
        self.resnet_block_2 = _Resnet_Conv_BN_ReLU(num_in_channels=64, num_out_channels=128, kernel_size=(5, 5), activation=activation)
        
        self.resnet_block_3 = _Resnet_Conv_BN_ReLU(num_in_channels=128, num_out_channels=256, kernel_size=(3, 3), activation=activation)
        self.resnet_block_4 = _Resnet_Conv_BN_ReLU(num_in_channels=256, num_out_channels=256, kernel_size=(3, 3), activation=activation)

        self.resnet_block_5 = _Resnet_Conv_BN_ReLU(num_in_channels=256, num_out_channels=128, kernel_size=(5, 5), activation=activation)
        self.resnet_block_6 = _Resnet_Conv_BN_ReLU(num_in_channels=128, num_out_channels=64, kernel_size=(5, 5), activation=activation)
        self.resnet_block_7 = _Resnet_Conv_BN_ReLU(num_in_channels=64, num_out_channels=32, kernel_size=(5, 5), activation=activation)

        self.output_conv = Conv2d(in_channels=32, out_channels=self.num_out_channels, kernel_size=(1, 1))

        self.upsample_1 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2)
        self.upsample_2 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2)
        self.upsample_3 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=2)

        self.__init_weight()

    def forward(self, x):

        # Encoder 
        en_conv1 = self.conv_block(x)
        pool1 = self.pool(en_conv1)
        
        en_conv2 = self.resnet_block_1(pool1)
        pool2 = self.pool(en_conv2)

        en_conv3 = self.resnet_block_2(pool2)
        pool3 = self.pool(en_conv3)
        

        # Bottleneck
        bottle1 = self.resnet_block_3(pool3)
        bottle2 = self.resnet_block_4(bottle1)
        
        # Decoder 
        upsample1 = self.upsample_1(bottle2)
        de_conv1 = self.resnet_block_5(torch.concat([upsample1, en_conv3], dim=1))

        upsample2 = self.upsample_2(de_conv1)
        de_conv2 = self.resnet_block_6(torch.concat([upsample2, en_conv2], dim=1))
       
        upsample3 = self.upsample_3(de_conv2)
        de_conv3 = self.resnet_block_7(torch.concat([upsample3, en_conv1], dim=1))

        if self.apply_final_layer:
            output = self.output_conv(de_conv3)
            return output
        else:
            return de_conv3
    
    def __init_weight(self):
        # weight initialize 
        for layer in self.modules():
            if isinstance(layer, Conv2d) or isinstance(layer, ConvTranspose2d):
                init_weights(layer)
                
        
    
    def load_trained_weights(self, weight_path):
        # load pretrained weights
        checkpoint = torch.load(weight_path)
        self.load_state_dict(checkpoint, strict=False)
    

    def freeze_encoder(self):
        # freeze encoder 
        Freeze_Layrers = ['conv_block', 'resnet_block_1', 'resnet_block_2']
        for name, layer in self.named_modules():
            name = name.split('.')[0]
            if name in Freeze_Layrers and isinstance(layer, Conv2d):
                for param in layer.parameters():
                    param.requires_grad = False
    
    def freeze_whole_model(self):
        # freeze the whole model
        for param in self.parameters():
            param.requires_grad = False


##########################################################################################################################
##############################                       CRF RES_UNet                        #################################
##########################################################################################################################



# class Res_UNet_CRF(Module):
#     # Res_UNet Model post-processed by CRF layer
#     def __init__(self, num_in_channels=3, num_out_channels=7, crf_num_iter=5):
#         # crf_num_iter: the number of iterations in CRF inference. 
#         # Remark: too large will cause gradient vanish and too small will not converge 
#         # Default is 5
#         super(Res_UNet_CRF, self).__init__()

#         self.base_model = Res_UNet(num_in_channels = num_in_channels, num_out_channels = num_out_channels, apply_final_layer=True)
#         self.CRF = CRF(n_ref=3, n_out=2, num_iter=crf_num_iter)

#     def forward(self, x):
#         output = self.base_model(x)
#         output = self.CRF(output, x)
#         return output


