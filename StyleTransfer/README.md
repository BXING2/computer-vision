## General
This example demonstrates training CycleGAN model for style transfer between two groups of images. 

## Dataset
The original dataset has around 1K Monet paintings and 7K natural photos. 1K paintings and 1K photos are used for training the model. The test dataset has 120 paints and 120 photos for evaluating the model performance. The image dimension is 256x256x3.

Dataset Link: https://www.kaggle.com/datasets/balraj98/monet2photo/data

## Model
The model is CycleGAN which consists of two GAN (Generative adversarial network) models. Each GAN model has one generator for converting images from one style to another, and one discriminator for judging if the images are real or fake. The adversarial loss is mse loss for generator and discriminator. In addition, the generator has a identity loss and cycle loss which is described by l1loss. 

## Evaluation
|Generator Loss|Discriminator Loss|
|---|---|
|<img src="figures/generator_loss.tif" /> | <img src="figures/discriminator_loss.tif" /> |

**Figure 1. Generator and discriminator loss on the train dataset at different epochs.**


<img src="figures/conf_matrix.tif" width="400" />

**Figure 2. Confusion matrix for 10 classes on the test dataset.**


| | Accuracy | Precison | Recall | F1 | 
| --- | --- | --- | --- | --- |
| Train | 0.987 | 0.986 | 0.986 | 0.986 |
| Validation | 0.955 | 0.954 | 0.954 | 0.954 |
| Test | 0.957 | 0.955 | 0.955 | 0.955 |

**Table 1. Summary of various metrics on train/validation/test dataset.**


Via finetuning the classification head, the model achieve an accuracy of 95.7% on the test dataset.

## Reference
1. https://huggingface.co/docs/transformers/main/en/model_doc/vit
2. Alexey, Dosovitskiy. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv: 2010.11929 (2020).
