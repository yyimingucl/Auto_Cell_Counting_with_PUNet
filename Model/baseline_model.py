import torch
from torch.nn import (Module,Conv2d, ReLU, MaxPool2d, \
                     Sequential, BatchNorm2d, UpsamplingBilinear2d, ConvTranspose2d, Identity)
from .Blocks import _Conv_BN_Activation, _Conv_BN_Activation_X2


# Define Models
class FCRN_A_BASE(Module):
    def __init__(self, model_type='base', num_in_channels=3, num_out_channels=7, activation='relu'):
        super(FCRN_A_BASE, self).__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        if model_type == 'base':
            self.conv_block = _Conv_BN_Activation
        elif model_type == 'v2':
            self.conv_block = _Conv_BN_Activation_X2

      
        self.conv_block_1 = self.conv_block(num_in_channels=self.num_in_channels, num_out_channels=32,
                        kernel_size=(3,3), activation=activation)
        self.conv_block_2 = self.conv_block(num_in_channels=32, num_out_channels=64,
                        kernel_size=(3,3), activation=activation)
        self.conv_block_3 = self.conv_block(num_in_channels=64, num_out_channels=128,
                        kernel_size=(3,3), activation=activation)
        self.conv_block_4 = self.conv_block(num_in_channels=128, num_out_channels=256,
                        kernel_size=(3,3), activation=activation)
        self.conv_block_5 = self.conv_block(num_in_channels=256, num_out_channels=128,
                        kernel_size=(3,3), activation=activation)
        self.conv_block_6 = self.conv_block(num_in_channels=128, num_out_channels=64,
                        kernel_size=(3,3), activation=activation)
        self.conv_block_7 = self.conv_block(num_in_channels=64, num_out_channels=32,
                        kernel_size=(3,3), activation=activation)
        self.conv_block_8 = self.conv_block(num_in_channels=32, num_out_channels=self.num_out_channels,
                        kernel_size=(3,3), activation=activation)

        self.MaxPooling = MaxPool2d(kernel_size=(2,2))   
        # self.UpSampling = UpsamplingBilinear2d(scale_factor=(2,2))
        self.UpSampling_1 = ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(2,2), stride=2)
        self.UpSampling_2 = ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2,2), stride=2)
        self.UpSampling_3 = ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2,2), stride=2)
        


    def forward(self, x):
        # Encoder
        en_conv1 = self.conv_block_1(x)
        pool1 = self.MaxPooling(en_conv1)
        # ==========================================
        en_conv2 = self.conv_block_2(pool1)
        pool2 = self.MaxPooling(en_conv2)
        # ==========================================
        en_conv3 = self.conv_block_3(pool2)
        pool3 = self.MaxPooling(en_conv3)
        # ==========================================
        conv4 = self.conv_block_4(pool3)
        # ==========================================
        # Decoder
        up1 = self.UpSampling_1(conv4)
        de_conv1 = self.conv_block_5(up1)
        # ==========================================
        up2 = self.UpSampling_2(de_conv1)
        de_conv2 = self.conv_block_6(up2)
        # ==========================================
        up3 = self.UpSampling_3(de_conv2)
        de_conv3 = self.conv_block_7(up3)
        # ==========================================
        output = self.conv_block_8(de_conv3)

        return output
    
    

            

class UNet(Module):
    def __init__(self, model_type='base', num_in_channels=3, num_out_channels=7, crf_num_iter=3, activation='relu'): #out = 7
        super(UNet, self).__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        if model_type == 'base':
            self.conv_block = _Conv_BN_Activation
        elif model_type == 'v2':
            self.conv_block = _Conv_BN_Activation_X2

        self.conv_block_1 = self.conv_block(num_in_channels=self.num_in_channels, num_out_channels=32, #32
                        kernel_size=(5,5), activation=activation)
        self.conv_block_2 = self.conv_block(num_in_channels=32, num_out_channels=64,   # 32 64
                        kernel_size=(5,5), activation=activation)
        self.conv_block_3 = self.conv_block(num_in_channels=64, num_out_channels=128,    # 64 128
                        kernel_size=(5,5), activation=activation)

        self.bottleneck_1 = self.conv_block(num_in_channels=128, num_out_channels=256,    # 128 256     
                        kernel_size=(3,3), activation=activation)
        self.bottleneck_2 = self.conv_block(num_in_channels=256, num_out_channels=256,    # 128 256     
                        kernel_size=(3,3), activation=activation)

        self.conv_block_5 =self.conv_block(num_in_channels=256, num_out_channels=128,     # 256 128          
                        kernel_size=(5,5), activation=activation)
        self.conv_block_6 = self.conv_block(num_in_channels=128, num_out_channels=64,    # 128*2 64
                        kernel_size=(5,5), activation=activation)
        self.conv_block_7 = self.conv_block(num_in_channels=64, num_out_channels=32,     # 64*2  32    
                        kernel_size=(5,5), activation=activation)

        self.output_conv = Conv2d(in_channels=32, out_channels=self.num_out_channels, kernel_size=(1, 1))

        self.MaxPooling = MaxPool2d(kernel_size=(2,2))   
        
        self.UpSampling_1 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2)
        self.UpSampling_2 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2)
        self.UpSampling_3 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=2)




    def forward(self, x):
        # Encoder
        en_conv1 = self.conv_block_1(x)
        pool1 = self.MaxPooling(en_conv1)
        # ==========================================
        en_conv2 = self.conv_block_2(pool1)
        pool2 = self.MaxPooling(en_conv2)
        # ==========================================
        en_conv3 = self.conv_block_3(pool2)
        pool3 = self.MaxPooling(en_conv3)
        # ==========================================
        
        bottle_1 = self.bottleneck_1(pool3)
        bottle_2 = self.bottleneck_2(bottle_1)

        # Decoder
        up1 = self.UpSampling_1(bottle_2)
        de_conv1 = self.conv_block_5(torch.cat((up1, en_conv3), dim=1))
        # ==========================================
        up2 = self.UpSampling_2(de_conv1)
        de_conv2 = self.conv_block_6(torch.cat((up2, en_conv2), dim=1))
        # ==========================================
        up3 = self.UpSampling_3(de_conv2)
        de_conv3 = self.conv_block_7(torch.cat((up3, en_conv1), dim=1))
        # ==========================================
        output = self.out_conv(de_conv3)

        return output