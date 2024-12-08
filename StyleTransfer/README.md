## General
This example demonstrates training CycleGAN model for style transfer between two groups of images. 

## Dataset
The original dataset has around 1K monet paintings and 7K natural photos. 1K paintings and 1K photos are used for training the model. The test dataset has 120 paintings and 120 photos for evaluating the model performance. The image dimension is 256x256x3.

Dataset Link: https://www.kaggle.com/datasets/balraj98/monet2photo/data

## Model
The model is CycleGAN which consists of two GAN (Generative adversarial network) models. Each GAN model has one generator for converting images from one style to another, and one discriminator for judging if the images are real or fake. The adversarial loss is mse loss for both generator and discriminator. In addition, the generator has an identity loss and a cycle loss which are described by l1loss. The model is trained for 20 epochs.

## Evaluation
|Generator Loss|Discriminator Loss|
|---|---|
|<img src="figures/generator_loss.tif" /> | <img src="figures/discriminator_loss.tif" /> |

**Figure 1. Generator and discriminator loss on the train dataset at different epochs.**

|FID for Monet paintings|FID for natural photos|
|---|---|
|<img src="figures/fid_fake_1.tif" /> | <img src="figures/fid_fake_2.tif" /> |

**Figure 2. FID scores on the test dataset for monet paintings (left) and natural photos (right) after different number of iterations. The dashed line represents the FID score between the real monet paintings and real natural photos.**

|Real Monet Paintings|Fake Natural Photos|
|---|---|
|<img src="figures/real_1.tif" /> | <img src="figures/fake_2.tif" /> |

**Figure 3. Examples of real monet paintings (left) and corresponding fake natural photos (right) generated from the GAN generator.**

|Real Natural Photos|Fake Monet Paintings|
|---|---|
|<img src="figures/real_2.tif" /> | <img src="figures/fake_1.tif" /> |

**Figure 4. Examples of real natural photos (left) and corresponding fake monet paintings (right) generated from the GAN generator.**


| Fake Natural Photo Generation | Fake Monet Painting Generation |
|---|---|
|<video src="https://github.com/user-attachments/assets/3022f4b4-bee9-4c99-a5a5-c0c950d07b24" ></video> | <video src=https://github.com/user-attachments/assets/5c73bb6a-8d42-4288-a3ca-b6b5abff6e19 ></video> |

**Video 1. Generated fake natural photos (left) and monet paintings (right) after different number of iterations.**

Figure 1 shows the generator and discriminator loss (both loss are averaged over two generators/discriminators). Figure 2 shows the FID scores for two groups of images after different number of iterations. For FID score of monet paintings, it is calculated from the embeddings (obtained from inception-v3 model) of real paintings and fake paintings generated from photos. For FID score of natural photos, it is calculated from the embeddings of real photos and fake photos generated from paintings. Both scores decrease as the number of iterations increases, indicating a increasing similarity between real and fake images. The black dashed line represents the FID score from the two groups of real images as a benchmark. Figure 3, 4 exhibits some examples of real paintings/fake photos and real photos/fake paintings. The video demonstrates how the generated fake images change as the number of iteration increases. 


## Reference
1. Goodfellow, Ian, et al. "Generative adversarial networks." Communications of the ACM 63.11 (2020): 139-144.
2. Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.

