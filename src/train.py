from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11n.pt")

results = model.train(
    data="voc.yaml", epochs=100, imgsz=640, batch=16, name="yolov8_voc", device=0
)
