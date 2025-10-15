import os
import glob
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import yaml
import random
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# ============================================================
# GPU CHECK
# ============================================================
def check_gpu():
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        print("CUDA available:", True)
        print("GPU detected:", torch.cuda.get_device_name(0))
    else:
        print("CUDA available:", False)
        print("Using CPU for training.")

# ============================================================
# PARSE CLASS NAMES
# ============================================================
def get_class_names(source_dir):
    print("Scanning dataset to generate class list...")
    xml_files = glob.glob(f"{source_dir}/**/*.xml", recursive=True)
    class_names = set()
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall("object"):
            cls = obj.find("name").text
            class_names.add(cls)
    class_names = sorted(list(class_names))
    print(f"Found {len(class_names)} unique classes:")
    print(class_names)
    return class_names

# ============================================================
# VOC → YOLO CONVERSION
# ============================================================
def convert_voc_to_yolo(xml_file, output_txt_path, class_list):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    with open(output_txt_path, "w") as f:
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in class_list:
                continue
            cls_id = class_list.index(cls)
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# ============================================================
# VISUALIZE YOLO LABELS
# ============================================================
def visualize_yolo_bboxes(image_path, label_path, class_names):
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if not os.path.exists(label_path):
        return

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h

        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_names[cls_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# ============================================================
# MAIN TRAINING ENTRY POINT
# ============================================================
if __name__ == "__main__":
    check_gpu()

    SOURCE_DIR = "../dataset/raw"
    OUTPUT_DIR = "../dataset/keyboard_yolo"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class_names = get_class_names(SOURCE_DIR)

    # Convert train/test
    for split in ["train", "test"]:
        img_dir = os.path.join(SOURCE_DIR, split, "images")
        xml_dir = os.path.join(SOURCE_DIR, split, "labels")

        output_img_dir = os.path.join(OUTPUT_DIR, split, "images")
        output_label_dir = os.path.join(OUTPUT_DIR, split, "labels")

        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for img_file in tqdm(files, desc=f"Converting {split} set"):
            xml_file = os.path.splitext(img_file)[0] + ".xml"
            xml_path = os.path.join(xml_dir, xml_file)
            if not os.path.exists(xml_path):
                continue

            shutil.copy(os.path.join(img_dir, img_file), os.path.join(output_img_dir, img_file))
            label_path = os.path.join(output_label_dir, os.path.splitext(img_file)[0] + ".txt")
            convert_voc_to_yolo(xml_path, label_path, class_names)

    print("XML to YOLO conversion completed for all splits.")

    # Visual check
    sample_images = glob.glob(f"{OUTPUT_DIR}/train/images/*")[:5]
    print("Previewing a few training samples with bounding boxes...")
    for img_path in sample_images:
        label_path = img_path.replace("\\images\\", "\\labels\\").rsplit(".", 1)[0] + ".txt"
        visualize_yolo_bboxes(img_path, label_path, class_names)

    # Create data.yaml
    data_config = {
        "train": os.path.abspath(f"{OUTPUT_DIR}/train/images"),
        "val": os.path.abspath(f"{OUTPUT_DIR}/test/images"),
        "nc": len(class_names),
        "names": class_names
    }
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        yaml.dump(data_config, f)
    print("data.yaml created.")

    # Train YOLOv8
    print("Starting YOLOv8 training on GPU if available...")
    model = YOLO("yolov8s.pt")
    results = model.train(
        data=f"{OUTPUT_DIR}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        workers=2,  # keep low for Windows
        name="keyboard_key_detector",
        project="runs/train",
        optimizer="SGD",
        lr0=0.01,
        freeze=10,
        device=0
    )

    metrics = model.val()
    print("Validation metrics:")
    print(metrics)

    model.export(format="onnx")
    print("Model exported to ONNX format.")
