#!/usr/bin/env python
# coding: utf-8

# import modules 
import os
import time
import numpy as np

import torch
import torchvision
from torchvision.transforms import v2

import data_utils
import model_utils

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import matplotlib.pyplot as plt

def main():

    # define device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # num of classes
    num_classes = 10 # 10 classes
    # use our dataset and defined transformations
    dataset = data_utils.Dataset("../data/EuroSAT", transforms=None)

    print(dataset.id2class.items())


    # split the dataset in train/valid/test parts
    torch.manual_seed(0) # set torch random seed
    indices = torch.randperm(len(dataset)).tolist()

    print(len(dataset.imgs), len(dataset.labels))

    print(indices[:10])
    train_split, valid_split, test_split = 0.6, 0.2, 0.2
    train_index = int(len(indices) * train_split)
    valid_index = int(len(indices) * valid_split) + train_index

    dataset_train = torch.utils.data.Subset(dataset, indices[: train_index])            # 800 * 0.6
    dataset_valid = torch.utils.data.Subset(dataset, indices[train_index: valid_index]) # 800 * 0.2
    dataset_test = torch.utils.data.Subset(dataset, indices[valid_index:])

    # define train/valid/test data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
    )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=32,
        shuffle=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False,
    )

    

    # load model
    model = model_utils.Model(num_classes)
    model = model.build_model().to(device)

    model.classifier.load_state_dict(torch.load("best_model_weights.pt"))


    # do prediction on test dataset

    truth, preds = [], []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader_test):

            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(
                          pixel_values=imgs,
                          labels=labels,
                          return_dict=True
                      )
            
            pred_labels = torch.argmax(outputs.logits, axis=1)
            
            # save truth and preds 
            truth += labels.cpu().numpy().tolist()
            preds += pred_labels.cpu().numpy().tolist()

    print(len(truth), len(preds))
    res = {"truth": truth, "preds": preds}
    np.save("res_test.npy", res)



if __name__ == "__main__":
    
    time_1 = time.time()
    main()
    time_2 = time.time()

    print(time_2 - time_1)
