# 
import os
import torch 
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from transformers import ViTImageProcessor

class Dataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms):
        
        # attributes  
        self.root = root
        self.transforms = transforms
         
        # build class2id/id2class maps
        self.id2class = {}
        self.class2id = {}
        for i, class_name in enumerate(sorted(os.listdir(root))):
            self.id2class[i] = class_name
            self.class2id[class_name] = i       
    
        # num of classes
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

        # load images and labels
        class_name = self.imgs[idx].split("_")[0]
        img_path = os.path.join(self.root, class_name, self.imgs[idx])

        img = read_image(img_path, mode=ImageReadMode.RGB)
        label = self.labels[idx]
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        img = self.processor.preprocess(img, return_tensors="pt")["pixel_values"].squeeze(dim=0)
        label = torch.tensor(label, dtype=torch.uint8).long()
    
        return img, label

def build_transforms(size, mean, std):
    
    # build transform pipeline
    transforms = []
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    transforms.append(v2.Resize(size=size, antialias=True))
    transforms.append(v2.Normalize(mean=mean, std=std))
    transforms = v2.Compose(transforms)
    
    return transforms
