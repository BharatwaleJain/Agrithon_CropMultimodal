import os
import shutil
from pathlib import Path
base_dir = "yolov8_insect_dataset"
train_img_dir = f"{base_dir}/images/train"
val_img_dir = f"{base_dir}/images/val"
train_label_dir = f"{base_dir}/labels/train"
val_label_dir = f"{base_dir}/labels/val"
for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    os.makedirs(dir_path, exist_ok=True)
source_dir = "obj_train_data"
image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
print(f"Found {len(image_files)} images")
train_files = image_files[:40]
val_files = image_files[40:]
for img_file in train_files:
    shutil.copy(os.path.join(source_dir, img_file), os.path.join(train_img_dir, img_file))
    label_file = img_file.replace('.jpg', '.txt')
    if os.path.exists(os.path.join(source_dir, label_file)):
        shutil.copy(os.path.join(source_dir, label_file), os.path.join(train_label_dir, label_file))
for img_file in val_files:
    shutil.copy(os.path.join(source_dir, img_file), os.path.join(val_img_dir, img_file))
    label_file = img_file.replace('.jpg', '.txt')
    if os.path.exists(os.path.join(source_dir, label_file)):
        shutil.copy(os.path.join(source_dir, label_file), os.path.join(val_label_dir, label_file))
print(f"Training set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")
print("YOLOv8 dataset structure created!")