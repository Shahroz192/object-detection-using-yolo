import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil

def convert_box(size, box):
    """Convert VOC bbox to YOLO format (x_center, y_center, width, height)"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # VOC format is [xmin, ymin, xmax, ymax]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    # Normalize
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return (x, y, w, h)

def convert_annotation(xml_file, output_path, class_dict):
    """Convert Pascal VOC xml file to YOLO txt file"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Open output file
    out_file = open(output_path, 'w')
    
    for obj in root.iter('object'):
        # Get class name
        cls = obj.find('name').text
        if cls not in class_dict:
            continue
        
        cls_id = class_dict[cls]
        
        # Get bounding box coordinates
        xmlbox = obj.find('bndbox')
        box = [
            float(xmlbox.find('xmin').text),
            float(xmlbox.find('ymin').text),
            float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymax').text)
        ]
        
        # Convert to YOLO format
        bb = convert_box((width, height), box)
        
        # Write to file
        out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
    
    out_file.close()

def process_voc_dataset(voc_path, output_path):
    """Process the entire Pascal VOC dataset"""
    # PASCAL VOC class names
    voc_classes = {
        'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
    }
    
    # Create output directories
    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
    
    # Get image sets
    years = ['2007', '2012']
    sets = ['train', 'val']  # Exclude 'test' if you want
    
    # Process each year and set
    for year in years:
        for image_set in sets:
            # Path to image sets file (train.txt, val.txt, etc.)
            image_set_file = os.path.join(voc_path, f'VOC{year}', 'ImageSets', 'Main', f'{image_set}.txt')
            
            if not os.path.exists(image_set_file):
                print(f"Warning: {image_set_file} does not exist. Skipping.")
                continue
                
            # Read image IDs
            with open(image_set_file, 'r') as f:
                image_ids = [line.strip() for line in f.readlines()]
            
            # Process each image
            for image_id in tqdm(image_ids, desc=f"Processing {year} {image_set}"):
                # XML annotation path
                xml_file = os.path.join(voc_path, f'VOC{year}', 'Annotations', f'{image_id}.xml')
                
                # Image file path
                image_file = os.path.join(voc_path, f'VOC{year}', 'JPEGImages', f'{image_id}.jpg')
                
                # Skip if files don't exist
                if not os.path.exists(xml_file) or not os.path.exists(image_file):
                    print(f"Warning: Missing files for {image_id}. Skipping.")
                    continue
                
                # Output paths
                output_dir = 'train' if image_set == 'train' else 'val'
                txt_output_path = os.path.join(output_path, 'labels', output_dir, f'{image_id}.txt')
                img_output_path = os.path.join(output_path, 'images', output_dir, f'{image_id}.jpg')
                
                # Convert annotation and copy image
                convert_annotation(xml_file, txt_output_path, voc_classes)
                shutil.copy(image_file, img_output_path)
    
    # Create YAML file for YOLOv8
    yaml_path = os.path.join(output_path, 'voc.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_path)}  # dataset root dir\n")
        f.write("train: images/train  # train images (relative to path)\n")
        f.write("val: images/val  # val images (relative to path)\n\n")
        
        f.write("# Classes\n")
        f.write("names:\n")
        for class_name, class_id in sorted(voc_classes.items(), key=lambda x: x[1]):
            f.write(f"  {class_id}: {class_name}\n")
    
    print(f"Conversion complete. Dataset prepared at {output_path}")
    print(f"Created YAML file at {yaml_path}")


process_voc_dataset("VOCdevkit", "voc")