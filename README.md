# Automatic Cell Counting in Digital Pathology

This repository is the pytorch implementation of the work **Automatic Cell Counting in Digital Pathology** and provides some baseline deep learning models for segmentation-based cell counting. 

## Repo Structure 
* data: folder contains the training data 
        + loss_weight_generator.py: provides function for producing loss weights.
* Model: modules for storing deep learning models
        + crfrnn_layer: modules for implementing Conditional Random Field on 2D image 
        + Blocks: contains different Convolutional Blocks and various util functions 
        + Resnet_based_model: contains Res_UNet, Res_UNet+CRF and probabilistic Res_UNet
        + Baseline: contains vanilla UNet
* loss.py: contains loss functions which could be jointly used with pixelwise loss weights and evaluation metrics. 
* train_basic.py: training script for baseline models 
* train_punet.py: training script for probabilistic UNet
* prediction.ipynb: visualization and make evaluations. 




