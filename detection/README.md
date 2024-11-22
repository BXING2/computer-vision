## General
This example demonstrates funetuning the YOLO model for object detection. 

## Dataset
The dataset is the Penn-Fudan Database which consists of 170 pedestrian images collected from University of Pennsylvania and Fudan University.

Dataset link: https://www.cis.upenn.edu/~jshi/ped_html/

## Model
The model is the YOLOv8 (You Only Look Once) with a detection head from Ultralytics. The weights in detection head are finetuned for 100 epoches, with all other model parameters in the backbone frozen.

## Evaluation
<img src="figures/map.png" width="400" />

**Figure 2. mAP50 and mAP50-95 on the training dataset at various epoches.**


| | Accuracy | Precison | Recall | F1 | 
| --- | --- | --- | --- | --- |
| Train | 0.987 | 0.986 | 0.986 | 0.986 |
| Validation | 0.955 | 0.954 | 0.954 | 0.954 |
| Test | 0.957 | 0.955 | 0.955 | 0.955 |

**Table 1. Summary of various metrics on train/validation/test dataset.**

<img src="figures/train_valid_loss.png" width="400" /> <img src="figures/train_valid_acc.png" width="400" />

**Figure 1. Loss and accuracy on the train and valiation dataset.**

Via finetuning the classification head, the model achieve an accuracy of 95.7% on the test dataset.

## Reference
1. https://huggingface.co/docs/transformers/main/en/model_doc/vit
2. Alexey, Dosovitskiy. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv: 2010.11929 (2020).