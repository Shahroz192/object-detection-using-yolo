from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/yolov8_voc/weights/best.pt")

# Path to test image
image_path = "hamster.jpg"
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        # Get confidence
        conf = box.conf[0]
        # Get class
        cls = box.cls[0]
        print(
            f"Detected {model.names[int(cls)]} with confidence {conf:.2f} at location {x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}"
        )

