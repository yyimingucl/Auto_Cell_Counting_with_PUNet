# Automatic Cell Counting in Digital Pathology

This repository is the pytorch implementation of the work **Automatic Cell Counting in Digital Pathology** and provides some baseline deep learning models for segmentation-based cell counting. 

## Set up Environments 
### 1. Docker 

All the experiemnts are running within the deepo docker image which is an all-in-one solution for ML/DL development and supports GPU acceleration. (more details on https://hub.docker.com/r/ufoym/deepo/)

* Step 1 pull image with ''' docker pull ufoym/deepo '''

* Step 2 run image with '''docker run --runtime=nvidia -i -p 8888:8888 -p 6060:22 --ipc=host --name="[Your Container Name]" -v [The Folder you wanna bind mount with container]:/data ufoym/deepo:latest jupyter'''

* Step 3 install further required packages

  - tensorboard = 2.8.0
  - scikit-image = 0.19.2
  - scipy = 1.8.0
  - seaborn = 0.11.2
  - opencv-python = 4.5.5.64
  - tqdm = 4.63.1



### 2. Conda run '''conda env create --name [Your env name] -f env.yml'''

## Repo Structure 
* data: folder contains the training data 
        + loss_weight_generator.py: provides function for producing loss weights.
* Model: modules for storing deep learning models

        + crfrnn_layer: modules for implementing Conditional Random Field on 2D image 

        + Blocks.py: contains different Convolutional Blocks and various util functions 

        + Res_UNet.py: contains Res_UNet 
                1.Res_UNet: Roberto Morelli et al. (2021): Automating cell counting in fuorescent microscopy through deep learning with c‑ResUnet                  (https://www.nature.com/articles/s41598-021-01929-5.pdf)
                2. CRF: Shuai Zheng et al. (2014): Conditional Random Fields as Recurrent Neural Networks (https://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf)
                CRF Implementation and Installation Guide: https://github.com/HapeMask/crfrnn_layer

        + Baseline.py: contains vanilla UNet and Fully Convolutional Regression Netwrok (FCRN)
                1. UNet: Olaf Ronneberger et al. (2015): U-Net: Convolutional Networks for Biomedical Image Segmentation (https://arxiv.org/abs/1505.04597)
                2. FCRN: Weidi Xie et al. (2015): Microscopy Cell Counting with Fully Convolutional Regression Networks (https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf)

        + Probabilistic_UNet.py: contains Probabilistic UNet 
                Probabilistic UNet: Simon A. A. Kohl et al. (2018): A Probabilistic U-Net for Segmentation of Ambiguous Images (https://arxiv.org/pdf/1806.05034.pdf)  Implementation: https://github.com/stefanknegt/Probabilistic-Unet-Pytorch

* loss.py: contains loss functions which could be jointly used with pixelwise loss weights and evaluation metrics. 
* train_basic.py: training script for baseline models 
* train_punet.py: training script for probabilistic UNet
* parameter.py: contains the meta parameters
* validation: folder for scripts to do validation for different model and different datasets
* post_process.py: marked watershed algorithm for post-processing generated masks. 


## Data Visualization 
#### CoNIC DataSet
![CoNIC DataSet](https://github.com/yyimingucl/Auto_Cell_Counting_with_PUNet/blob/main/readme_image/datavisualization.png)


