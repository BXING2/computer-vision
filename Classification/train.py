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

    '''
    for name, params in model.named_parameters():
        print(name, params.shape, params.requires_grad)
    '''

    optimizer = torch.optim.Adam( # SGD(
        #params,
        model.parameters(),
        lr=0.0001,
        #momentum=0.9,
        #weight_decay=0.0005
    )

    # build learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        min_lr=1e-6,
    )

    # train model
    num_epochs = 50
    best_valid_loss = float("inf") # best validation loss 

    metric = {}
    metric["train_loss"] = []
    metric["valid_loss"] = []
    metric["valid_acc"] = []

    for epoch in range(num_epochs):

        train_loss = 0
        # train 
        model.train()
        for i, batch in enumerate(data_loader_train):

            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(
                          pixel_values=imgs,
                          labels=labels,
                          return_dict=True
                      )

            model.zero_grad()
            outputs.loss.backward()
            optimizer.step()

            train_loss += outputs.loss.item()

        train_loss /= len(data_loader_train)

        valid_loss, valid_acc = 0, 0
        # valid
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader_valid):

                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(
                              pixel_values=imgs,
                              labels=labels,
                              return_dict=True
                          )
                
                valid_loss += outputs.loss.item()

                pred_labels = torch.argmax(outputs.logits, axis=1)
                acc = torch.sum(pred_labels == labels) / len(labels)
                valid_acc += acc.item()

            valid_loss /= len(data_loader_valid)
            valid_acc /= len(data_loader_valid)

        # update saved model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            # save model weight
            print("best model epoch {}".format(epoch))
            torch.save(model.classifier.state_dict(), "best_model_weights.pt")

        # save metrics
        metric["train_loss"].append(train_loss)
        metric["valid_loss"].append(valid_loss)
        metric["valid_acc"].append(valid_acc)

        # update the learning rate
        lr_scheduler.step(valid_loss)

        print(train_loss, valid_loss, valid_acc)

    np.save("metric.npy", metric)



if __name__ == "__main__":
    
    time_1 = time.time()
    main()
    time_2 = time.time()

    print(time_2 - time_1)
