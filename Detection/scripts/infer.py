from ultralytics import YOLO

# params
#model = YOLO("yolov8n.pt") 
model = YOLO("result/weights/best.pt")


# model evaluation

for split in ["train", "val", "test"]:
    metric = model.val(
        data="dataset.yaml",
        batch=16,
        workers=2,
        iou=0.5,
        split=split,
    )

    print(split, metric)
