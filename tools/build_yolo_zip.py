# -*- coding: utf-8 -*-
import os
import sys
import json
import shutil
import zipfile
import random

# Configuration
MAX_IMAGES = 1000
IMG_WIDTH = 644
IMG_HEIGHT = 600
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15  # 15% test data

# Base directory
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = script_dir

# Find the most recent "recording_" folder
recording_folders = sorted([
    f for f in os.listdir(base_dir)
    if f.startswith("recording_") and os.path.isdir(os.path.join(base_dir, f))
])
if not recording_folders:
    print("No 'recording_' folder found!")
    sys.exit(1)

recording_folder = os.path.join(base_dir, recording_folders[-1])
img_dir = os.path.join(recording_folder, "img")
labels_json = os.path.join(recording_folder, "labels.json")

# Prepare target structure
dataset_dir = os.path.join(script_dir, "dataset")
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# Load labels
with open(labels_json, "r") as f:
    raw_data = json.load(f)

valid_data = []
for entry in raw_data:
    frame = entry["frame"]
    x1, y1, w, h = entry["bbox"]

    # Skip invalid bounding boxes
    if x1 < 0 or y1 < 0 or x1 + w > IMG_WIDTH or y1 + h > IMG_HEIGHT:
        continue

    # Convert to YOLO format (normalized center x, y, width, height)
    xc = (x1 + w / 2.0) / IMG_WIDTH
    yc = (y1 + h / 2.0) / IMG_HEIGHT
    norm_w = w / IMG_WIDTH
    norm_h = h / IMG_HEIGHT

    if all(0.0 <= v <= 1.0 for v in [xc, yc, norm_w, norm_h]):
        entry["yolo_label"] = f"0 {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}\n"
        valid_data.append(entry)

print(f"Filtered: {len(valid_data)} valid entries")

# Shuffle and limit data
random.seed(42)
random.shuffle(valid_data)
if MAX_IMAGES:
    valid_data = valid_data[:MAX_IMAGES]

# Train/val/test split
val_count = int(VAL_SPLIT * len(valid_data))
test_count = int(TEST_SPLIT * len(valid_data))
train_count = len(valid_data) - val_count - test_count

train_data = valid_data[:train_count]
val_data = valid_data[train_count:train_count+val_count]
test_data = valid_data[train_count+val_count:]

def write_split(data_split, split_name):
    count = 0
    for entry in data_split:
        frame = entry["frame"]
        label_line = entry["yolo_label"]

        img_src = os.path.join(img_dir, f"{frame}.jpg")
        img_dst = os.path.join(images_dir, split_name, f"{frame}.jpg")
        label_path = os.path.join(labels_dir, split_name, f"{frame}.txt")

        try:
            shutil.copyfile(img_src, img_dst)
            with open(label_path, "w") as f:
                f.write(label_line)
            count += 1
        except Exception as e:
            print(f"Error with frame {frame}: {e}")
    return count

v_train = write_split(train_data, "train")
v_val = write_split(val_data, "val")
v_test = write_split(test_data, "test")

print(f"Train: {v_train} images with labels")
print(f"Val:   {v_val} images with labels")
print(f"Test:  {v_test} images with labels")

# Write data.yaml
yaml_path = os.path.join(dataset_dir, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""train: images/train
val: images/val
test: images/test

nc: 1
names: ['ball']
""")

# Create ZIP archive
zip_path = os.path.join(script_dir, "dataset.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for foldername, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            arcname = os.path.relpath(filepath, dataset_dir)
            zipf.write(filepath, arcname)

print(f"YOLO dataset ZIP created: {zip_path}")
