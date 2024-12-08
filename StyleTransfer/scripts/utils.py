
import os
import glob
from PIL import Image

import numpy as np

import torch
from torch import nn


class Dataset(torch.utils.data.Dataset):
    '''
    Class for loading data
    '''

    def __init__(
        self,
        root, # root directory of the dataset
        transform=False, # transform pipeline
        mode="train", # train or test
    ):

        # list of two set of image files  
        self.file_list_1 = sorted(glob.glob(os.path.join(root, f"{mode}A/*")))
        self.file_list_2 = sorted(glob.glob(os.path.join(root, f"{mode}B/*")))

        # make sure two groups have the same amount of files 
        size_1, size_2 = len(self.file_list_1), len(self.file_list_2)
        if size_1 < size_2: # list 1 has few files
            self.file_list_2 = [self.file_list_2[index] for index in np.random.permutation(size_2)[:size_1].astype(np.int32)]
        else: # list 2 has fewer files
            self.file_list_1 = [self.file_list_1[index] for index in np.random.permutation(size_1)[:size_2].astype(np.int32)]

        self.transform = transform
        
        print("data size", min(size_1, size_2))
        print(len(self.file_list_1), len(self.file_list_2))

    def __len__(self):
        
        return len(self.file_list_1)

    def __getitem__(self, idx):

        img_1 = self.transform(Image.open(self.file_list_1[idx]))
        img_2 = self.transform(Image.open(self.file_list_2[idx]))

        return img_1, img_2



class Generator(nn.Module):
    '''
    Class for building generator of GAN model
    '''

    def __init__(
        self,
        input_feats, # number of channels of input image 
        output_feats, # number of channels of output image
        hidden_feats=64, # number of hidden channels
        n_residual_blocks=0, # number of residual blocks
    ):
        super(Generator, self).__init__()
        
        self.input_feats = input_feats
        self.output_feats = output_feats
        self.hidden_feats = hidden_feats
        self.n_residual_blocks = n_residual_blocks

        
        # initial block
        self.initial_conv = nn.Conv2d(
            in_channels=input_feats,
            out_channels=hidden_feats,
            kernel_size=7,
            padding=3,
        ) # WxWx3 -> WxWxn

        # down path
        self.down1 = DownBlock(
            input_feats=hidden_feats,
        ) # WxWxn -> W/2xW/2x2n
        self.down2 = DownBlock(
            input_feats=hidden_feats * 2,
        ) # W/2xW/2x2n -> W/4xW/4x4n
        
        # residual block 
        self.res = ResidualBlock(
            input_feats=hidden_feats * 4,
        ) # W/4xW/4x4n -> W/4xW/4x4n
        
        # up path
        self.up1 = UpBlock(
            input_feats=hidden_feats * 4,
        ) # W/4xW/4x4n -> W/2xW/2x2n
        self.up2 = UpBlock(
            input_feats=hidden_feats * 2,
        ) # W/2xW/2x2n -> WxWxn
   
        
        # final block
        self.final_conv = nn.Conv2d(
            in_channels=hidden_feats,
            out_channels=input_feats,
            kernel_size=7,
            padding=3,
        ) # WxWxn -> WxWx3


    def forward(self, x):
        
        x0 = self.initial_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        x3 = self.res(x2)
        for _ in range(self.n_residual_blocks):
            x3 = self.res(x3)

        x4 = self.up1(x3)
        x5 = self.up2(x4)
        x6 = self.final_conv(x5)
        
        return nn.Tanh()(x6) # convert image to (-1, 1)



class Discriminator(nn.Module):
    '''
    Class for building discriminator for GAN model
    '''
    def __init__(
        self,
        input_feats, # number of input channels
        hidden_feats=64,
    ):
        super(Discriminator, self).__init__()
   
        # initial block
        self.initial_conv = nn.Conv2d(
            in_channels=input_feats,
            out_channels=hidden_feats,
            kernel_size=7,
            padding=3,
        ) # WxWx3 -> WxWxn
        
        self.down1 = DownBlock(
            hidden_feats,
            # kernel_size=3, #4,
        ) # WxWxn -> W/2xW/2x2n

        self.down2 = DownBlock(
            hidden_feats * 2,
            # kernel_size=3, #4,
        ) # W/2xW/2x2n -> W/4xW/4x4n

        self.down3 = DownBlock(
            hidden_feats * 4,
            # kernel_size=3, #4,
        ) # W/4xW/4x4n -> W/8xW/8x8n

        # final block
        self.final_conv = nn.Conv2d(hidden_feats * 8, 1, kernel_size=1)

    def forward(self, x):
        
        x0 = self.initial_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.final_conv(x3)

        return x4



class ResidualBlock(nn.Module):
    def __init__(self, input_feats):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_feats,
            out_channels=input_feats,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=input_feats,
            out_channels=input_feats,
            kernel_size=3,
            padding=1,
        )

        self.instancenorm = nn.InstanceNorm2d(num_features=input_feats)
        self.activation = nn.ReLU()

    def forward(self, x):
        
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        
        return original_x + x


class DownBlock(nn.Module):
    # reduce image spatial dimension and double number of channels 
    def __init__(
        self,
        input_feats, # input channels of image
    ):
        super(DownBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_feats,
            out_channels=input_feats * 2,  # double number of channels 
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.instancenorm = nn.InstanceNorm2d(
            num_features=input_feats * 2,
        )

        self.activation = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x):
        # print(x.shape) 
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.pool(x) 
       
        return x

class UpBlock(nn.Module):
    # increase image spatial dimensiona and reduce number of channels 
    def __init__(
        self,
        input_feats, # input channels of images
    ):
        
        super(UpBlock, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(
            in_channels=input_feats,
            out_channels=input_feats // 2, # reduce number of channels by 2
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1)

        self.instancenorm = nn.InstanceNorm2d(input_feats // 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        
        return x

