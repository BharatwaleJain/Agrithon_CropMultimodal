import os
import json
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2
import numpy as np
def convert_coco_to_yolo_seg(coco_json_path, images_dir, output_dir):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    train_img_dir = f"{output_dir}/images/train"
    val_img_dir = f"{output_dir}/images/val"
    train_label_dir = f"{output_dir}/labels/train"
    val_label_dir = f"{output_dir}/labels/val"
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    image_info = {img['id']: img['file_name'] for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    image_files = list(image_info.values())
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    process_split(coco_data, image_info, annotations_by_image, train_files, images_dir, train_img_dir, train_label_dir, "train")
    process_split(coco_data, image_info, annotations_by_image, val_files, images_dir, val_img_dir, val_label_dir, "val")
    print("Dataset conversion completed!")
def process_split(coco_data, image_info, annotations_by_image, file_list, src_img_dir, dst_img_dir, dst_label_dir, split_name):
    filename_to_id = {v: k for k, v in image_info.items()}
    for filename in file_list:
        src_img_path = os.path.join(src_img_dir, filename)
        dst_img_path = os.path.join(dst_img_dir, filename)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
            image = cv2.imread(src_img_path)
            img_height, img_width = image.shape[:2]
            image_id = filename_to_id[filename]
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(dst_label_dir, label_filename)
            with open(label_path, 'w') as f:
                if image_id in annotations_by_image:
                    for ann in annotations_by_image[image_id]:
                        if 'segmentation' in ann and ann['segmentation']:
                            class_id = 0
                            segmentation = ann['segmentation'][0]
                            normalized_seg = []
                            for i in range(0, len(segmentation), 2):
                                x = segmentation[i] / img_width
                                y = segmentation[i + 1] / img_height
                                normalized_seg.extend([x, y])
                            seg_str = ' '.join([f'{coord:.6f}' for coord in normalized_seg])
                            f.write(f"{class_id} {seg_str}\n")
            print(f"Processed {split_name}: {filename}")
if __name__ == "__main__":
    coco_json_path = "annotations/instances_default.json"
    images_dir = "images/default"
    output_dir = "yolov8_disease_dataset"
    convert_coco_to_yolo_seg(coco_json_path, images_dir, output_dir)