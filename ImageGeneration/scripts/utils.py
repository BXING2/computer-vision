import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision import models, transforms


class Dataset(torch.utils.data.Dataset):
    '''
    Class for loading data
    '''
 
    def __init__(
        self,
        root,            # dataset directory
        transform=False, # transform pipeline
        mode="train",    # train or test
    ):

        if mode == "train":
            self.dataset = torchvision.datasets.CIFAR10(
                root=root,
                train=True,
                download=True,
            )

        if mode == "test":
            self.dataset = torchvision.datasets.CIFAR10(
                root=root,
                train=False,
                download=True,
            )

        print(f"dataset size: {len(self.dataset)}")

        self.transform = transform

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        # image and label
        image, label = self.dataset[idx]

        # transform image
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).to(torch.int64)
        label = nn.functional.one_hot(label, num_classes=10).to(torch.float32)

        return image, label



class UNet(nn.Module):
    '''
    UNet model structure
    '''
    def __init__(
        self,
        input_dims,        # input spatial dims
        input_feats,       # input feature dims
        hidden_feats=128,  # hidden feature dims
    ):  
  
        super(UNet, self).__init__()

        self.input_dims = input_dims
        self.input_feats = input_feats
        self.hidden_feats = hidden_feats

        # initial conv layer
        self.init_conv = ResidualBlock(input_feats, hidden_feats)  # WxWx3 -> WxWxn

        # down path 
        self.down1 = UNetDown(hidden_feats, hidden_feats)  # WxWxn -> W/2xW/2xn 
        self.down2 = UNetDown(hidden_feats, hidden_feats * 2) # W/2xW/2xn -> W/4xW/4x2n

        self.down3 = nn.Sequential(
            nn.AvgPool2d(self.input_dims // 4),
            #nn.GELU(),
            nn.ReLU(),
        ) # W/8xW/8x4n -> 1x1x4n

        
        # timestep embedding learnt from neural network  
        #self.time_embed1 = Embed(1, hidden_feats * 2)
        #self.time_embed2 = Embed(1, hidden_feats * 1)


        # fixed sin/cos time embeddings
        self.time_embed1 = TimeEmbed(hidden_feats * 2)
        self.time_embed2 = TimeEmbed(hidden_feats * 1)


        # context embeddings learnt from neural network 
        self.c_embed1 = Embed(10, hidden_feats * 2)
        self.c_embed2 = Embed(10, hidden_feats * 1)
        
        # up path
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_feats * 2, hidden_feats * 2, self.input_dims // 4, self.input_dims // 4),
            #nn.InstanceNorm2d(hidden_feats * 2),
            nn.LayerNorm([hidden_feats * 2, input_dims // 4, input_dims // 4]), 
            #nn.BatchNorm2d(hidden_feats * 2),
            #nn.GroupNorm(self.input_dims // 4, hidden_feats * 2),
            #nn.GELU(),
            nn.ReLU(),
        ) # 1x1x4n -> W/8xW/8x4n

        self.up2 = UNetUp(hidden_feats * 4, hidden_feats * 1) # W/8xW/8x4n(2) -> W/4xW/4x2n 
        self.up3 = UNetUp(hidden_feats * 2, hidden_feats) # W/4xW/4x2n(2) -> W/2xW/2xn
        # self.up4 = UNetUp(hidden_feats * 2, hidden_feats) # W/4xW/4x2n(2) -> W/2xW/2xn

        # final conv layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_feats * 2, hidden_feats, 3, 1, 1),
            #nn.InstanceNorm2d(hidden_feats),
            nn.LayerNorm([hidden_feats, self.input_dims, self.input_dims]), 
            #nn.BatchNorm2d(hidden_feats),
            #nn.GroupNorm(self.input_dims // 4, hidden_feats), # normalize
            #nn.GELU(),
            nn.ReLU(),
            nn.Conv2d(hidden_feats, self.input_feats, 3, 1, 1),
        )   # WxWxn(2) -> WxWx3

    def forward(
        self,
        x, # images
        t, # timesteps
        c, # contexts
    ):
       
        # initial conv  
        x = self.init_conv(x)
        
        # down path 
        down1 = self.down1(x)       #[10, 256, 8, 8]
        down2 = self.down2(down1)   #[10, 256, 4, 4]
        down3 = self.down3(down2)
        

        time_embed1 = self.time_embed1(t)[:, :, None, None]
        time_embed2 = self.time_embed2(t)[:, :, None, None]

        c_embed1 = self.c_embed1(c).view(-1, self.hidden_feats * 2, 1, 1)
        c_embed2 = self.c_embed2(c).view(-1, self.hidden_feats * 1, 1, 1)

        # up path
        up1 = self.up1(down3)
        up2 = self.up2(c_embed1 * up1 + time_embed1, down2)  # add and multiply embeddings
        up3 = self.up3(c_embed2 * up2 + time_embed2, down1)  # 

        # final conv
        output = self.final_conv(torch.cat((up3, x), 1))

        '''
        # check intermediate image dims
        print("x:", x.shape)
        print("down path: down1/down2/down3/down4", down1.shape, down2.shape, down3.shape, down4.shape)
        print("time embedding 1/2:", time_embed1.shape, time_embed2.shape, time_embed3.shape)
        print("up path: up1/up2/up3", up1.shape, up2.shape, up3.shape, up4.shape)
        print("output:", output.shape)
        '''        

        return output


class ConvBlock(nn.Module):

    def __init__(
        self,
        input_feats,
        output_feats,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats, 3, 1, 1),  # number of features changes 
            #nn.InstanceNorm2d(output_feats), 
            nn.BatchNorm2d(output_feats),
            #nn.GELU(),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(output_feats, output_feats, 3, 1, 1),
            #nn.InstanceNorm2d(output_feats),
            nn.BatchNorm2d(output_feats),
            #nn.GELU(),
            nn.ReLU(),
        )

    def forward(
        self,
        x,
    ):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        return x2

class ResidualBlock(nn.Module):

    def __init__(
        self,
        input_feats,
        output_feats,
    ):
        super().__init__()

        self.input_feats = input_feats
        self.output_feats = output_feats

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats, 3, 1, 1),  # number of features changes 
            #nn.InstanceNorm2d(output_feats), 
            nn.BatchNorm2d(output_feats),
            #nn.GELU(),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(output_feats, output_feats, 3, 1, 1),
            #nn.InstanceNorm2d(output_feats),
            nn.BatchNorm2d(output_feats),
            #nn.GELU(),
            nn.ReLU(),
        )

    def forward(
        self,
        x,
    ):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        # if input and output have the same number of features 
        if self.input_feats == self.output_feats:
            return (x + x2) / 1.414

        shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x)
        
        return (shortcut(x) + x2) / 1.414


class UNetUp(nn.Module):
    def __init__(
        self,
        input_feats,
        output_feats,
    ):
        super(UNetUp, self).__init__()

        layers = [
            nn.ConvTranspose2d(input_feats, output_feats, 2, 2),
            #ConvBlock(output_feats, output_feats),
            #ConvBlock(output_feats, output_feats),
            ResidualBlock(output_feats, output_feats),
            ResidualBlock(output_feats, output_feats),
        ]

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        x,      # data
        skip,   # data from UNetDown 
    ):

        # combine data along feature dimension
        x = torch.cat((x, skip), 1)

        return self.model(x)


class UNetDown(nn.Module):
    def __init__(
        self,
        input_feats,
        output_feats,
    ):
        super(UNetDown, self).__init__()

        layers = [
            #ConvBlock(input_feats, output_feats),
            ResidualBlock(input_feats, output_feats),
            ResidualBlock(output_feats, output_feats),
            nn.MaxPool2d(2),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        return self.model(x)



class Embed(nn.Module):
    def __init__(
        self,
        input_dims,
        embed_dims,
    ):
        super(Embed, self).__init__()

        self.input_dims = input_dims

        layers = [
            nn.Linear(input_dims, embed_dims),
            #nn.GELU(),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dims)

        return self.model(x)

class TimeEmbed(nn.Module):
    def __init__(
        self,
        dims,
        n=10000,
    ):

        super(TimeEmbed, self).__init__()    

        self.dims = dims
        self.n = n

        self.denom = n ** (torch.arange(dims) / dims)

    def forward(
        self,
        t, 
    ):

        embeddings = t[:, None] / self.denom[None, :].to(t.device)

        # sin component
        embeddings[::2] = embeddings[::2].sin()
        embeddings[1::2] = embeddings[1::2].cos()

        return embeddings
