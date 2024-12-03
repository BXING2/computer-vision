from ultralytics import YOLO

# params
data_config_path = <PATH> # 

# load yolo model
model = YOLO("yolov8n.pt")

# train model
model.train(
    data=data_config_path,
    epochs=100, #200,
    batch=16,
    workers=2,
    #freeze=22,   # freeze all layers except for the final layer
    freeze=10,    # freeze backbone
    optimizer="AdamW",
    lr0=3e-5,
)

