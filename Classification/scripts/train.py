import os
import time
import numpy as np

import torch

import utils

def train():
    '''
    Pipeline for training ViT model for image classification
    '''

    # --- load device --- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # --- load dataset --- #

    n_classes = 10 # 10 classes
    batch_size = 32
    file_name = "../../data/EuroSAT"

    dataset = utils.Dataset(
        file_name,
    )

    # train/valid/test subset
    torch.manual_seed(0) # set torch random seed
    indices = torch.randperm(len(dataset)).tolist()

    train_split, valid_split, test_split = 0.6, 0.2, 0.2
    train_index = int(len(indices) * train_split)
    valid_index = int(len(indices) * valid_split) + train_index

    dataset_train = torch.utils.data.Subset(dataset, indices[: train_index])
    dataset_valid = torch.utils.data.Subset(dataset, indices[train_index: valid_index]) 
    dataset_test = torch.utils.data.Subset(dataset, indices[valid_index:])

    # train/valid/test data loaders
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

    

    # --- load model --- #
    model = utils.load_model(
        n_classes,
    ).to(device)

    for name, params in model.named_parameters():
        print(name, params.shape, params.requires_grad)


    # --- load optimizer --- #
    optim = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )


    
    # --- learning rate scheduler --- #
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim,
        mode="min",
        min_lr=1e-6,
    )



    # --- train model --- #
    n_epochs = 50
    best_valid_loss = float("inf") # best validation loss 

    metric = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
    }
    
    for epoch in range(n_epochs):

        # train 
        train_loss, train_acc = 0, 0
        model.train()
        for i, batch in enumerate(data_loader_train):

            # move data to cuda
            batch = {key: val.to(device) for key, val in batch.items()}
            
            # forward process  
            outputs = model(
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                return_dict=True
            )
                
            # backward process
            model.zero_grad()
            outputs.loss.backward()
            optim.step()

            # accumulate train loss
            train_loss += outputs.loss.item()

            # accumulate train accuracy
            pred_labels = torch.argmax(outputs.logits, axis=1)
            acc = torch.sum(pred_labels == batch["labels"]) / len(pred_labels)
            train_acc += acc.item()
        
        train_loss /= len(data_loader_train)
        train_acc /= len(data_loader_train)


        # validation
        valid_loss, valid_acc = 0, 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader_valid):
                
                # move data to cuda 
                batch = {key: val.to(device) for key, val in batch.items()}

                # forward process
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                    return_dict=True
                )
                
                # accumulate validation loss
                valid_loss += outputs.loss.item()
            
                # accumulate validation accuracy 
                pred_labels = torch.argmax(outputs.logits, axis=1)
                acc = torch.sum(pred_labels == batch["labels"]) / len(pred_labels)
                valid_acc += acc.item()

            valid_loss /= len(data_loader_valid)
            valid_acc /= len(data_loader_valid)

        # update saved model if validation loss decreases
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            # save model weight
            print("best model epoch {}".format(epoch))
            torch.save(model.classifier.state_dict(), "best_model_weights.pt")

        # save metrics
        metric["train_loss"].append(train_loss)
        metric["train_acc"].append(train_acc)
        metric["valid_loss"].append(valid_loss)
        metric["valid_acc"].append(valid_acc)

        # update the learning rate
        lr_scheduler.step(valid_loss)

        print(train_loss, train_acc, valid_loss, valid_acc)

    np.save("train_valid_metric.npy", metric)


def main():
    train()

if __name__ == "__main__":
    
    time_1 = time.time()
    main()
    time_2 = time.time()

    print(time_2 - time_1)
