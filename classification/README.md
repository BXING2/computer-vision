## General
This example mainly demonstrates the process of funetuning vision transformer model for classification tasks. 

## Dataset
The dataset is from Torchvison (https://pytorch.org/vision/main/generated/torchvision.datasets.EuroSAT.html) which consists of satellite land images with ten classes.

## Model
The model is the vision transformer (ViT) with a classification head (ViTForImageClassification). The classification layer weights are finetuned for 50 epoches, with all other model parameters frozen.

## Evaluation
![Alt text](figures/train_valid_loss.png)
<img src="figures/train_valid_acc.png" width="300" />


## Reference
