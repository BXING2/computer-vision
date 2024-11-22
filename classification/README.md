## General
This example mainly demonstrates the process of funetuning vision transformer model for classification tasks. 

## Dataset
The dataset is from Torchvison (https://pytorch.org/vision/main/generated/torchvision.datasets.EuroSAT.html) which consists of satellite land images with ten classes.

## Model
The model is the vision transformer (ViT) with a classification head (ViTForImageClassification). The classification layer weights are finetuned for 50 epoches, with all other model parameters frozen.

## Evaluation
<img src="figures/train_valid_loss.png" width="400" /> <img src="figures/train_valid_acc.png" width="400" />

**Figure 1. Loss and accuracy on the train and valiation dataset.**


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
