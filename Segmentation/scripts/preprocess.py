# converting bounding box format from XYXY to XYWH

import numpy as np
from PIL import Image

import os
import glob
import shutil

from torchvision.io import read_image

# 

def convert_format(
    orig_file_path=None,
    box_file_path=None,
    mask_file_path=None,
    ):

    # load image file and box file names
    boxes = glob.glob(os.path.join(orig_file_path, "Annotation/*"))

    for box in boxes:

        # get file basename 
        basename = os.path.basename(box)

        # load data
        with open(box) as f:
            lines = f.readlines()
        lines = [line.strip().split() for line in lines]

        # write file 
        saved_box_path = os.path.join(box_file_path, basename[:-3] + "txt")
        saved_mask_path = os.path.join(mask_file_path, basename[:-3] + "txt")
        f_box = open(saved_box_path, "w")
        f_mask = open(saved_mask_path, "w")

        # image width and height
        img_width = float(lines[2][-5]) # X
        img_height = float(lines[2][-3]) # Y
      
        # get bounding box coordinates 
        for i in range(10, len(lines), 5):
            # get xlim and ylim
            xmin = float(lines[i][-5].strip("(),"))
            ymin = float(lines[i][-4].strip("(),"))
            xmax = float(lines[i][-2].strip("(),"))
            ymax = float(lines[i][-1].strip("(),"))

            # convert bounding box format 
            x_center = 0.5 * (xmin + xmax)
            y_center = 0.5 * (ymin + ymax)

            scaled_x_center = x_center / img_width
            scaled_y_center = y_center / img_height

            scaled_box_width = (xmax - xmin) / img_width
            scaled_box_height = (ymax - ymin) / img_height

            # write to target folder
            print("{} {} {} {} {}".format(0, scaled_x_center, scaled_y_center, scaled_box_width, scaled_box_height), file=f_box)
        
        # get instance coordinates
        mask = read_image(os.path.join(orig_file_path, "PedMasks", basename[:-4]+"_mask.png")).squeeze()
        instance_ids = np.unique(mask)
        for instance_id in instance_ids:
            if instance_id == 0:
                continue
            coords = np.where(mask == instance_id) # first dim: y, second dim: x
            scaled_y = coords[0] / img_height
            scaled_x = coords[1] / img_width 
            
            info = ["0"] 
            for x, y in zip(scaled_x, scaled_y):
                info.extend([str(x.round(6)), str(y.round(6))])

            info = " ".join(info)
            print(info, file=f_mask)  

        f_box.close()
        f_mask.close()

    return 

def split_data(
    orig_image_path,
    orig_label_path,
    targ_path,
    ):

    # count number of files in the label folder
    basenames = [os.path.basename(label_file) for label_file in glob.glob(os.path.join(orig_label_path, "*"))]
    np.random.shuffle(basenames)
    size = len(basenames)


    # train/valid/test  index
    train_index = int(size * 0.6)
    valid_index = int(size * 0.2)
    
    train_labels, valid_labels, test_labels = np.split(
                                                  ary=basenames,
                                                  indices_or_sections=[train_index, train_index+valid_index])
    
    # remove and build new directory
    if os.path.isdir(targ_path):
        shutil.rmtree(targ_path)
    
    for name in ["train", "valid", "test"]:
        os.makedirs(os.path.join(targ_path, name, "images",))
        os.makedirs(os.path.join(targ_path, name, "labels",))


    # build train/valid/test dataset following yolov8 format
    for labels, split in zip([train_labels, valid_labels, test_labels], ["train", "valid", "test"]):
        for label in labels:
            image = label[:-3] + "png"

            shutil.copy(
                src=os.path.join(orig_label_path, label),
                dst=os.path.join(targ_path, split, "labels",),
                )

            shutil.copy(
                src=os.path.join(orig_image_path, image),
                dst=os.path.join(targ_path, split, "images",),
                )
            
    return 

#task = "convert"
task = "split"


if task == "convert":

    # params 
    orig_file_path = "../../ped_data/PennFudanPed"
    box_file_path = "box"
    mask_file_path = "mask"

    convert_format(
        orig_file_path,
        box_file_path,
        mask_file_path,
        )


if task == "split":

    # params
    orig_image_path = "../../ped_data/PennFudanPed/PNGImages/"
    orig_label_path = "mask/"
    targ_path = "data/"

    split_data(
        orig_image_path,
        orig_label_path,
        targ_path,
        )
