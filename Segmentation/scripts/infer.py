from ultralytics import YOLO

# params
#model = YOLO("yolov8n-seg.pt") 
model = YOLO("result/weights/best.pt")  # load a pretrained model (recommended for training)


# model inference

for split in ["train", "val", "test"]:
    metric = model.val(
        data="dataset.yaml",
        batch=16,
        workers=2,
        iou=0.5,
        split=split,
    )

    print(split)
    print(metric.results_dict)

