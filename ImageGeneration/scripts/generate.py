import numpy as np

import torch
import torch.nn as nn

import utils 

from torchvision.transforms import Resize

def generate():
    '''
    Pipeline for generating images from diffusion model
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

    # sample settings
    n_batches = 20
    batch_size = 500

    # --- construct noise schedule --- #   
    # beta at different timesteps 
    beta_t = (beta_T - beta_0) * torch.linspace(0, 1, n_timesteps + 1, device=device) + beta_0
    # alpha at different timesteps 
    alpha_t = torch.cumsum((1 - beta_t).log(), dim=0).exp()
    alpha_t[0] = 1



    # --- load model  --- # 
    model = utils.UNet(
        input_dims=input_dims,
        input_feats=input_feats,
        hidden_feats=hidden_feats,
    ).to(device)

    # load model weights
    #model_weight_path = "final_weights.pt"
    model_weight_path = "weights.pt"
    model.load_state_dict(torch.load(model_weight_path, weights_only=True,))
    model.eval()


    # resize images if needed
    transform = Resize((128, 128))


    # --- generate fake images --- #
    fake_images = []
    for index in range(n_batches):
       
        # initial noises 
        samples = torch.randn(batch_size, input_feats, input_dims, input_dims).to(device)  
        
        # sample random classes 
        labels = torch.randint(0, 10, (len(samples),)).to(torch.int64)
        labels = torch.nn.functional.one_hot(labels, num_classes=10).to(torch.float32).to(device)
        print(samples.shape, labels.shape)


        with torch.no_grad(): 
            for i in range(n_timesteps, 0, -1): # step can change
                if i % 100 == 0: print("timestep {}".format(i))

                # timestep
                #t = torch.tensor([i / n_timesteps])[:, None, None, None].to(device)
                t = torch.tensor([i] * len(samples)).to(device)

                # pred noises
                pred_noises = model(samples, t, labels)

                # new noises 
                new_noises = torch.randn_like(samples) if i > 1 else 0

                # images after one-step denoising             
                samples = 1.0 / (1 - beta_t[i]).sqrt() * (samples - beta_t[i] / (1 - alpha_t[i]).sqrt() * pred_noises) + beta_t[i].sqrt() * new_noises

        # store fake images
        fake_images.append(samples.detach().cpu().numpy())

    fake_images = np.concatenate(fake_images, axis=0)

    print(fake_images.shape)

    # swap spatial and channel dimension  
    fake_images = np.transpose(fake_images, [0, 2, 3, 1])
    
    # convert pixel values to 0-255
    x_min = fake_images.min(axis=(1, 2), keepdims=True)   
    x_max = fake_images.max(axis=(1, 2), keepdims=True)
    
    fake_images = (fake_images - x_min) / (x_max - x_min) * 255.

    print(fake_images.shape)
    
    # save images
    np.save("fake_images.npy", fake_images)

def main():
    generate()

if __name__ == "__main__":
    main()
