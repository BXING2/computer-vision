import os
import time
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms


import utils, loss_utils

def train():
    '''
    Pipeline for training CycleGAN model
    '''

    # ---- define params --- # 
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

    data_path = "../../data" 
    input_feats = 3
    
    n_epochs = 20
    batch_size = 5
    learning_rate = 1e-4
    save_frequency = 5



    # --- load dataset --- #
    transform = torchvision.transforms.Compose([
        transforms.ToTensor(), # from 0-255 -> 0-1
        transforms.Normalize((0.5,), (0.5,)), # from 0-1 -> -1-1
    ])

    dataset = utils.Dataset(
        root=data_path,
        transform=transform,
        mode="train",
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )


    
    # --- load CycleGAN model --- # 
    # CycleGAN has two GAN models, each of which has a generator and discriminator

    # generator
    generator_1to2 = utils.Generator(input_feats, input_feats, n_residual_blocks=3).to(device) # generator converting 1 to 2
    generator_2to1 = utils.Generator(input_feats, input_feats, n_residual_blocks=3).to(device) # generator converting 2 to 1

    # discriminator 
    discriminator_1 = utils.Discriminator(input_feats).to(device) # discriminator for 1 
    discriminator_2 = utils.Discriminator(input_feats).to(device) # discriminator for 2
    


    # --- load optimizer --- # 

    # generator optimizer
    generator_optim = torch.optim.Adam(
        list(generator_1to2.parameters()) + list(generator_2to1.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    # discriminator 1 optimizer
    discriminator_1_optim = torch.optim.Adam(
        discriminator_1.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    # discriminator 2 optimizer
    discriminator_2_optim = torch.optim.Adam(
        discriminator_2.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )


    # --- load loss --- # 

    # adversarial loss for generator/discriminator
    adversarial_loss = nn.MSELoss()
    
    # identity/cycle loss
    reconstruction_loss = nn.L1Loss() 
   


    # ---  train loops --- #
 
    loss = []
   
    n_iterations = 0 
    for epoch in range(n_epochs):
        curr_generator_loss, curr_discriminator_loss = 0, 0
        for real_1, real_2 in dataloader:
       
            n_iterations += 1
            curr_batch_size = len(real_1)
            
            # move data to device
            real_1 = real_1.to(device)
            real_2 = real_2.to(device)
            

            ##### update discriminator

            # generate fake images 
            with torch.no_grad():
                fake_1 = generator_2to1(real_2)
                fake_2 = generator_1to2(real_1)
                # print(fake_1.shape, fake_2.shape)

            # compute discriminator 1 loss
            discriminator_1_loss = loss_utils.compute_discriminator_loss(
                real_1,
                fake_1,
                discriminator_1,
                adversarial_loss,
            )

            # compute discriminator 2 loss
            discriminator_2_loss = loss_utils.compute_discriminator_loss(
                real_2,
                fake_2,
                discriminator_2,
                adversarial_loss,
            )
        
            # backpropagation of discriminators 
            discriminator_1_optim.zero_grad()
            discriminator_2_optim.zero_grad()

            discriminator_1_loss.backward(retain_graph=True)
            discriminator_2_loss.backward(retain_graph=True)

            discriminator_1_optim.step()
            discriminator_2_optim.step()
            ##### discriminator update finished 


            ##### update generator

            # compute generator loss
            generator_loss, fake_1, fake_2 = loss_utils.compute_generator_loss(
                real_1,
                real_2,
                generator_1to2,
                generator_2to1,
                discriminator_1,
                discriminator_2,
                adversarial_loss,    # adversarial loss 
                reconstruction_loss, # identity loss 
                reconstruction_loss, # cycle consistency loss 
            )
            
            # backpropagation of generator
            generator_optim.zero_grad()
            generator_loss.backward()
            generator_optim.step()

            ##### generator update finished 


            # accumulate generator/discriminator loss
            curr_generator_loss += generator_loss.item()
            curr_discriminator_loss += discriminator_1_loss.item() + discriminator_2_loss.item()
    
 
            # save model every number of epochs 
            # if epoch == 0 or (epoch+1) % save_frequency == 0:
            if epoch <= 1:
                if n_iterations % 50 == 0:        

                    torch.save({
                        "generator_1to2": generator_1to2.state_dict(),
                        "generator_2to1": generator_2to1.state_dict(),
                        #"discriminator_1": discriminator_1.state_dict(),
                        #"discriminator_2": discriminator_2.state_dict(),
                    }, f"cycleGAN_i_{n_iterations}.pt")  # epoch+1

            else:
                if (epoch+1) % 2 == 0:

                    torch.save({
                        "generator_1to2": generator_1to2.state_dict(),
                        "generator_2to1": generator_2to1.state_dict(),
                        #"discriminator_1": discriminator_1.state_dict(),
                        #"discriminator_2": discriminator_2.state_dict(),
                    }, f"cycleGAN_e_{epoch+1}.pt")  # epoch+1

        curr_generator_loss /= len(dataloader)
        curr_discriminator_loss /= len(dataloader)
        loss.append([curr_generator_loss, curr_discriminator_loss])   
 
    np.save("GAN_loss.npy", loss)

    # save final model
    torch.save({
        "generator_1to2": generator_1to2.state_dict(),
        "generator_2to1": generator_2to1.state_dict(),
    }, f"cycleGAN_final.pt")  # epoch+1


def main():
    train()

if __name__ == "__main__":

    time_1 = time.time()
    main()
    time_2 = time.time()

    print("Total running time: {} s".format(time_2 - time_1))

