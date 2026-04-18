# Object Detection Using YOLO

Real-time object detection using YOLO (You Only Look Once) on VOC dataset with 20 object classes.

## 🎯 Project Overview

| Component | Description |
|-----------|-------------|
| **Model** | YOLO (You Only Look Once) |
| **Dataset** | Pascal VOC |
| **Classes** | 20 (person, car, dog, cat, etc.) |
| **Framework** | Ultralytics YOLO |

## 📁 Project Structure

```
object-detection-using-yolo/
├── src/
│   ├── train.py         # Training script
│   ├── inference.py     # Inference/prediction
│   ├── yolo_format.py   # VOC to YOLO conversion
│   └── main.py          # Main entry point
├── data/
│   └── download_dataset.sh  # Dataset download script
├── models/              # Trained weights
├── runs/                # Training runs/outputs
├── notebooks/           # EDA notebooks
├── voc.yaml            # Dataset configuration
└── data.yaml           # YOLO config
```

## 🛠️ Tech Stack

- **Python** ≥3.8
- **Deep Learning**: PyTorch
- **Object Detection**: Ultralytics YOLO
- **Dataset**: Pascal VOC

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/Shahroz192/object-detection-using-yolo.git
cd object-detection-using-yolo

# Install dependencies
pip install ultralytics torch

# Download dataset
bash data/download_dataset.sh

# Train model
python src/train.py

# Run inference
python src/inference.py --image path/to/image.jpg
```

## 📊 Supported Classes

| ID | Class |
|----|-------|
| 0 | aeroplane |
| 1 | bicycle |
| 2 | bird |
| 3 | boat |
| 4 | bottle |
| 5 | bus |
| 6 | car |
| 7 | cat |
| 8 | chair |
| 9 | cow |
| 10 | diningtable |
| 11 | dog |
| 12 | horse |
| 13 | motorbike |
| 14 | person |
| 15 | pottedplant |
| 16 | sheep |
| 17 | sofa |
| 18 | train |
| 19 | tvmonitor |

## 🔧 Usage

### Training
```bash
python src/train.py --epochs 100 --batch 16
```

### Inference
```bash
python src/inference.py --source images/ --save
```

## 📈 Results

- mAP (mean Average Precision) tracking via YOLO
- Training metrics saved to `runs/`
- Best weights saved in `models/`

## 🐳 Docker

```bash
docker build -t yolo-detection .
docker run yolo-detection python src/inference.py
```

## 📝 License

MIT License

---

*Computer Vision project - part of ML portfolio*