#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yiming Yang
# Date: 08/07/2022
# Email: y.yang2@napier.ac.uk
# Description: define the different convolutional blocks and functions for building neural networks


from torch.nn import (Module,Conv2d, ReLU, Sequential, BatchNorm2d, Identity, ELU, ConvTranspose2d)
from torch.nn.init import kaiming_normal_, orthogonal_

def _Conv_BN_Activation(num_in_channels, num_out_channels, kernel_size, 
                 padding_mode='zeros', stride=1, activation='relu'):
    if activation == 'relu':
        activation = ReLU(inplace=True)
    elif activation == 'elu':
        activation = ELU(inplace=True)
    elif activation == None:
        activation = Identity()
    else:
        raise TypeError('{} is Unknown Activation Function'.format(activation))

    Conv2D_Block = Sequential(
        Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,kernel_size=kernel_size, 
               padding='same', stride=stride,padding_mode=padding_mode),
        BatchNorm2d(num_features = num_out_channels),
        activation
    )
    return Conv2D_Block

def _Conv_BN_Activation_X2(num_in_channels, num_out_channels, kernel_size, 
                 padding_mode='zeros', stride=1, activation='relu'):
    if activation == 'relu':
        activation = ReLU(inplace=True)
    elif activation == 'elu':
        activation = ELU(inplace=True)
    elif activation == None:
        activation = Identity()
    else:
        raise TypeError('{} is Unknown Activation Function'.format(activation))

    Conv2D_Block_X2 = Sequential(
        Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,kernel_size=kernel_size, 
               padding='same', stride=stride,padding_mode=padding_mode),
        BatchNorm2d(num_features = num_out_channels),
        activation,
        Conv2d(in_channels=num_out_channels, out_channels=num_out_channels,kernel_size=kernel_size, 
               padding='same', stride=stride,padding_mode=padding_mode),
        BatchNorm2d(num_features = num_out_channels),
        activation
    )
    return Conv2D_Block_X2


class _Resnet_Conv_BN_ReLU(Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size, padding_mode='zeros', stride=1, activation='relu'):
        super(_Resnet_Conv_BN_ReLU, self).__init__()

        self.conv_11 = Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,kernel_size=(1,1), 
                              padding='same', stride=stride, padding_mode=padding_mode)

        self.conv_block_1 = _Conv_BN_Activation_X2(num_in_channels, num_out_channels, kernel_size, 
                                                padding_mode, stride, activation)
        self.conv_block_2 = _Conv_BN_Activation(num_out_channels, num_out_channels, kernel_size, 
                                                padding_mode, stride, activation=None)
        self.elu = ELU(inplace=True)

    def forward(self, x):
        identity = self.conv_11(x)

        out = self.conv_block_1(x)
        out = self.conv_block_2(out)

        out += identity
        out = self.elu(out)

        return out  

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == Conv2d or type(m) == ConvTranspose2d:
        kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == Conv2d or type(m) == ConvTranspose2d:
        orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

