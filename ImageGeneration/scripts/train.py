import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import models, transforms

import utils



def train():
    '''
    Pipeline for training diffusion model for image generations 
    '''

    # --- define params --- #

    # load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # diffusion settings
    n_timesteps = 500
    beta_0, beta_T = 1e-4, 0.02 

    # model settings
    input_dims = 32
    input_feats = 3
    hidden_feats = 128

    # training settings
    n_epochs = 200
    batch_size = 100
    learning_rate = 1e-3


    # --- construct noise schedule --- #   
    # beta at different timesteps 
    beta_t = (beta_T - beta_0) * torch.linspace(0, 1, n_timesteps + 1, device=device) + beta_0
    # alpha at different timesteps 
    alpha_t = torch.cumsum((1 - beta_t).log(), dim=0).exp()
    alpha_t[0] = 1



    # --- load dataset --- # 
    
    # path of dataset  
    root = "../../data"
    
    # pipeline for transforming data 
    transform = transforms.Compose([
        #RandomHorizontalFlip(),
        #RandomRotation((-45, 45)),
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
    ])
   

    # load train/valid data 
    train_valid_dataset = utils.Dataset(
        root,
        transform,
        mode="train",
    )
    # load test data
    test_dataset = utils.Dataset(
        root,
        transform,
        mode="test",
    )
    
    # train/valid/test subset
    torch.manual_seed(0) # set torch random seed
    indices = torch.randperm(len(train_valid_dataset)).tolist()

    train_split, valid_split = 0.9, 0.1
    train_index = int(len(indices) * train_split)

    train_dataset = torch.utils.data.Subset(train_valid_dataset, indices[: train_index])
    valid_dataset = torch.utils.data.Subset(train_valid_dataset, indices[train_index:]) 

    print(len(train_dataset), len(valid_dataset))

    # define train/valid/test data loaders
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    data_loader_valid = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    

    # --- load model  --- # 
    model = utils.UNet(
        input_dims=input_dims,
        input_feats=input_feats,
        hidden_feats=hidden_feats,
    ).to(device)

    
    print(model)

    # --- load optimizer --- # 
    optim = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )


    # --- load learning rate scheduler --- # 
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim,
        mode="min",
        min_lr=1e-6,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs, eta_min=1e-6)

    # --- train model --- # 

    best_valid_loss = float("inf")
    f = open("log.txt", "w")
    
    for epoch in range(n_epochs):

        # optim.param_groups[0]['lr'] = 1e-3 * (1 - epoch / n_epochs)

        # train
        train_loss = 0
        model.train()

        for i, batch in enumerate(data_loader_train):
            
            # move data to device 
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)           
            # print(images.shape, labels.shape)
            # print(label)
            # sample noises
            noises = torch.randn_like(images)
            
            # sample timesteps
            timesteps = torch.randint(1, n_timesteps + 1, (len(images),)).to(device)
            
            # add noises to images 
            new_images = alpha_t[timesteps, None, None, None].sqrt() * images + (1 - alpha_t[timesteps, None, None, None]).sqrt() * noises

            # predict noises
            pred_noises = model(
                new_images,
                #timesteps / n_timesteps,  # timesteps are normalized by the total timesteps
                timesteps, 
                labels,
            )        

            # loss is mean squared error between the predicted and true noise
            loss = nn.MSELoss(pred_noises, noises)          

            # backward process 
            optim.zero_grad()
            loss.backward()
            optim.step()
            

            # accumulate train loss
            train_loss += loss.item()       
 
        train_loss /= len(data_loader_train)

        # validate
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader_valid):
            
                # move data to device
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)                

                # sample noises and timesteps 
                noises = torch.randn_like(images)
                timesteps = torch.randint(1, n_timesteps + 1, (len(images),)).to(device)
                # print(images.shape, noises.shape, timesteps.shape)
                # add noises to images 
                new_images = alpha_t[timesteps, None, None, None].sqrt() * images + (1 - alpha_t[timesteps, None, None, None]).sqrt() * noises

                # predict noises
                pred_noises = model(
                    new_images,
                    #timesteps / n_timesteps,
                    timesteps, # absolute timesteps for sin position encoding 
                    labels,
                )
    

                # compute loss 
                loss = F.mse_loss(pred_noises, noises)

                # accumulate valiation loss
                valid_loss += loss.item()
            
            valid_loss /= len(data_loader_valid)

        print("epoch {} train loss {} valid loss {}".format(epoch, train_loss, valid_loss))

        
        # update saved model as validation loss decreases 
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            # save model weight
            print("best model epoch {}".format(epoch))
            torch.save(model.state_dict(), "weights.pt")
        
 
        # update the learning rate
        #lr_scheduler.step(valid_loss)
        lr_scheduler.step()

        # print train/validation loss
        print("{} {}".format(train_loss, valid_loss), file=f)
    
    torch.save(model.state_dict(), "final_weights.pt")

    # close log files
    f.close()


def main():
    train()

if __name__ == "__main__":

    time_1 = time.time()
    main()
    time_2 = time.time()

    print("Total running time: {} s".format(time_2 - time_1))
