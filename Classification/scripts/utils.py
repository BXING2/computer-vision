# 
import os
import torch 
from torchvision.io import read_image, ImageReadMode
from transformers import ViTImageProcessor, ViTForImageClassification

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
    ):
        
        self.root = root
         
        # class2id/id2class maps
        self.id2class = {}
        self.class2id = {}
        for i, class_name in enumerate(sorted(os.listdir(root))):
            self.id2class[i] = class_name
            self.class2id[class_name] = i       
    
        # number of classes
        self.num_classes = len(self.id2class)

        # load image file names and labels 
        self.imgs, self.labels = [], []
        for class_id, class_name in self.id2class.items():
            curr_imgs = sorted([name for name in os.listdir(os.path.join(root, class_name))])
            self.imgs += curr_imgs
            self.labels += [class_id] * len(curr_imgs) 
        
        # set up vit image processor
        self.processor = ViTImageProcessor(
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
        )         

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, idx):

        # load one image
        class_name = self.imgs[idx].split("_")[0]
        img_path = os.path.join(
            self.root,
            class_name,
            self.imgs[idx],
        )

        img = read_image(
            img_path,
            mode=ImageReadMode.RGB,
        )

        # load one label
        label = self.labels[idx]
        
        # process image and label 
        img = self.processor.preprocess(
            img,
            return_tensors="pt",
        )["pixel_values"].squeeze(dim=0)

        label = torch.tensor(
            label,
            dtype=torch.uint8,
        ).long()
    
        # return img, label
        return  {
            "pixel_values": img, 
            "labels": label,
        }


def load_model(
    n_classes,
):


    # load ViT model
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )

    # freeze bert layers for fine tuning
    for params in model.vit.parameters():
        params.requires_grad = False

    return model

