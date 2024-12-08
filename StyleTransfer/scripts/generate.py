import os
import time
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms 

import utils, loss_utils

def generate():
    '''
    Piple for generating images using CycleGAN
    '''

    # ---- define params --- # 

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    data_path = "../../data"
    input_feats = 3
    batch_size = 4


    # --- load dataset --- #
    transform = torchvision.transforms.Compose([
        transforms.ToTensor(), # from 0-255 -> 0-1
        transforms.Normalize((0.5,), (0.5,)), # from 0-1 -> -1-1
    ])

    dataset = utils.Dataset(
        root=data_path,
        transform=transform,
        mode="test",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )



    # --- load generators of CycleGAN
    generator_1to2 = utils.Generator(input_feats, input_feats, n_residual_blocks=3).to(device) # generator converting 1 to 2
    generator_2to1 = utils.Generator(input_feats, input_feats, n_residual_blocks=3).to(device) # generator converting 2 to 1
   


    # --- generate images --- #
    # save images from models at various levels 
    
    epochs = range(4, 21, 2)
    iterations = range(50, 401, 50)

    # epochs = [1]

    reals = {"real_1": [], "real_2": []}    
    
    #epoch = -1
    #for iteration in iterations:
    for epoch in epochs:
 
        fakes = {"fake_1": [], "fake_2": []}
        
        # load generator weights
        group_weights = torch.load("weights/cycleGAN_e_{}.pt".format(epoch), weights_only=False)
        #group_weights = torch.load("weights/cycleGAN_i_{}.pt".format(iteration), weights_only=False)
        generator_1to2.load_state_dict(group_weights["generator_1to2"])
        generator_2to1.load_state_dict(group_weights["generator_2to1"])
     
        # set GAN as eval mode
        generator_1to2.eval()
        generator_2to1.eval() 
    
        # implement image style transfer
        for real_1, real_2 in dataloader:
            
            # load samples from real images 
            real_1 = real_1.to(device)
            real_2 = real_2.to(device)

            # generate fake images 
            with torch.no_grad():
                
                fake_1 = generator_2to1(real_2).permute((0, 2, 3, 1))
                fake_2 = generator_1to2(real_1).permute((0, 2, 3, 1))

                # normalize fake images
                fake_1_min, fake_1_max = torch.amin(fake_1, dim=(1, 2), keepdim=True), torch.amax(fake_1, dim=(1, 2), keepdim=True)
                fake_2_min, fake_2_max = torch.amin(fake_2, dim=(1, 2), keepdim=True), torch.amax(fake_2, dim=(1, 2), keepdim=True)
                
                # print(fake_1.shape, fake_2.shape, fake_1_min.shape, fake_1_max.shape) 

                fake_1 = (fake_1 - fake_1_min) / (fake_1_max - fake_1_min)
                fake_2 = (fake_2 - fake_2_min) / (fake_2_max - fake_2_min)

                # save fake results
                fakes["fake_1"].append(fake_1.cpu().numpy())
                fakes["fake_2"].append(fake_2.cpu().numpy())


                # whether save real images
                if epoch == 4:            

                    # normalize real images
                    real_1 = real_1.detach().permute((0, 2, 3, 1))
                    real_2 = real_2.detach().permute((0, 2, 3, 1))

                    real_1_min, real_1_max = torch.amin(real_1, dim=(1, 2), keepdim=True), torch.amax(real_1, dim=(1, 2), keepdim=True)
                    real_2_min, real_2_max = torch.amin(real_2, dim=(1, 2), keepdim=True), torch.amax(real_2, dim=(1, 2), keepdim=True)
                    real_1 = (real_1 - real_1_min) / (real_1_max - real_1_min)            
                    real_2 = (real_2 - real_2_min) / (real_2_max - real_2_min) 

                    # save real images
                    reals["real_1"].append(real_1.cpu().numpy())
                    reals["real_2"].append(real_2.cpu().numpy())
        

        if epoch == 4:
            reals["real_1"] = np.concatenate(reals["real_1"], axis=0) 
            reals["real_2"] = np.concatenate(reals["real_2"], axis=0) 
            
            # save data to hard disk
            np.save("real_pairs.npy", reals)
        
        fakes["fake_1"] = np.concatenate(fakes["fake_1"], axis=0)       
        fakes["fake_2"] = np.concatenate(fakes["fake_2"], axis=0)       

        #np.save("fake_pairs_i_{}.npy".format(iteration), fakes)
        np.save("fake_pairs_e_{}.npy".format(epoch), fakes)
 
        for key, val in fakes.items():
            print(key, val.shape)


def main():
    time_s = time.time()
    generate()
    time_e = time.time()

    print("INFERENCE TIME {}".format(time_e-time_s))

if __name__ == "__main__":
    main()
    
