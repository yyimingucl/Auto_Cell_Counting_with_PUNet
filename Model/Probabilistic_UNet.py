#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yiming Yang
# Date: 08/07/2022
# Email: y.yang2@napier.ac.uk
# Description: Define the Probabilistic UNet Model.
#              Referenced from Simon A. A. Kohl et al. (2018): A Probabilistic U-Net for Segmentation of Ambiguous Images
#              Implementation: https://github.com/stefanknegt/Probabilistic-Unet-Pytorch

import torch
import numpy as np
from torch.nn import (Module,Conv2d, Conv2d, init, ModuleList, Sequential)
from torch.distributions import Normal, Independent, kl
from .Blocks import _Conv_BN_Activation, _Resnet_Conv_BN_ReLU, init_weights, init_weights_orthogonal_normal
from .Res_UNet import Res_UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet Encoder
class Encoder(Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, activation='relu', padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = ModuleList()
        self.input_channels = input_channels
        

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1
      
        layers = []

        conv_block = _Conv_BN_Activation(num_in_channels=self.input_channels, num_out_channels=32, kernel_size=(3, 3), activation=activation)       
        resnet_block_1 = _Resnet_Conv_BN_ReLU(num_in_channels=32, num_out_channels=64, kernel_size=(3, 3), activation=activation)
        resnet_block_2 = _Resnet_Conv_BN_ReLU(num_in_channels=64, num_out_channels=128, kernel_size=(3, 3), activation=activation)
        
        layers = [conv_block, resnet_block_1, resnet_block_2]

        self.layers = Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        # self.num_filters = num_filters
        # self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, posterior=self.posterior)
        self.conv_layer = Conv2d(128, 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        # encoding = torch.mean(encoding, dim=2, keepdim=True)
        # encoding = torch.mean(encoding, dim=3, keepdim=True)

        encoding = torch.mean(encoding, dim=(2,3), keepdim=True)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist

class Fcomb(Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim,  num_classes, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.name = 'Fcomb'


        if self.use_tile:

            resnet_block_1 = _Resnet_Conv_BN_ReLU(num_in_channels=32+self.latent_dim, num_out_channels=32, kernel_size=(1, 1))
            resnet_block_2 = _Resnet_Conv_BN_ReLU(num_in_channels=32, num_out_channels=16, kernel_size=(1, 1))
            conv_block_1 = _Conv_BN_Activation(num_in_channels=16, num_out_channels=self.num_classes, kernel_size=(1, 1))

            self.layers = [resnet_block_1, resnet_block_2]
            self.last_layer = conv_block_1

            self.layers = Sequential(*self.layers)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticUnet(Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, num_in_channels, num_classes, latent_dim, criterion, beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = num_in_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta
        self.z_prior_sample = 0
        self.criterion = criterion
        

        self.unet = Res_UNet(self.input_channels).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.latent_dim, self.initializers).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.latent_dim, self.initializers, posterior=True).to(device)
        self.fcomb = Fcomb(self.input_channels, self.latent_dim, self.num_classes, initializers={'w':'orthogonal', 'b':'normal'}, use_tile=True).to(device)

    def forward(self, patch, segm=None, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            #z_prior = self.prior_latent_space.base_dist.loc 
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features,z_prior)


    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False, loss_weight=None):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        # criterion = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)
        z_posterior = self.posterior_latent_space.rsample()
        
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)
        reconstruction_loss = self.criterion(preds=self.reconstruction, targets=segm, pixelwise_weights=loss_weight)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return -(self.reconstruction_loss + self.beta * self.kl)