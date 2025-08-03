import albumentations as A
import cv2
import os
import numpy as np
import shutil
def read_yolo_bbox(label_path, img_height, img_width):
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    x_min = (x_center - width/2) * img_width
                    y_min = (y_center - height/2) * img_height
                    x_max = (x_center + width/2) * img_width
                    y_max = (y_center + height/2) * img_height
                    bboxes.append([x_min, y_min, x_max, y_max, class_id])
    return bboxes
def write_yolo_bbox(bboxes, label_path, img_height, img_width):
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            if len(bbox) >= 5:
                x_min, y_min, x_max, y_max, class_id = bbox
                x_center = (x_min + x_max) / (2 * img_width)
                y_center = (y_min + y_max) / (2 * img_height)
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
transform = A.Compose([
    A.Rotate(limit=30, p=0.8, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
input_images_dir = 'yolov8_insect_dataset/images/train'
input_labels_dir = 'yolov8_insect_dataset/labels/train'
output_images_dir = 'augmented_insect_dataset/images/train'
output_labels_dir = 'augmented_insect_dataset/labels/train'
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)
os.makedirs('augmented_insect_dataset/images/val', exist_ok=True)
os.makedirs('augmented_insect_dataset/labels/val', exist_ok=True)
for img_name in os.listdir(input_images_dir):
    if img_name.endswith('.jpg'):
        shutil.copy(os.path.join(input_images_dir, img_name), output_images_dir)
        label_name = img_name.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(input_labels_dir, label_name)):
            shutil.copy(os.path.join(input_labels_dir, label_name), output_labels_dir)
shutil.copytree('yolov8_insect_dataset/images/val', 'augmented_insect_dataset/images/val', dirs_exist_ok=True)
shutil.copytree('yolov8_insect_dataset/labels/val', 'augmented_insect_dataset/labels/val', dirs_exist_ok=True)
num_augmentations = 3
for img_name in os.listdir(input_images_dir):
    if not img_name.endswith('.jpg'):
        continue
    img_path = os.path.join(input_images_dir, img_name)
    label_name = img_name.replace('.jpg', '.txt')
    label_path = os.path.join(input_labels_dir, label_name)
    image = cv2.imread(img_path)
    img_height, img_width = image.shape[:2]
    bboxes = read_yolo_bbox(label_path, img_height, img_width)
    for aug_idx in range(num_augmentations):
        try:
            class_labels = [bbox[4] for bbox in bboxes]
            bbox_coords = [bbox[:4] for bbox in bboxes]
            augmented = transform(image=image, bboxes=bbox_coords, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']
            aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{aug_idx}.jpg"
            aug_img_path = os.path.join(output_images_dir, aug_img_name)
            cv2.imwrite(aug_img_path, aug_image)
            aug_label_name = f"{os.path.splitext(img_name)[0]}_aug_{aug_idx}.txt"
            aug_label_path = os.path.join(output_labels_dir, aug_label_name)
            final_bboxes = [[*bbox, label] for bbox, label in zip(aug_bboxes, aug_labels)]
            write_yolo_bbox(final_bboxes, aug_label_path, aug_image.shape[0], aug_image.shape[1])
        except Exception as e:
            print(f"Error augmenting {img_name}: {e}")
print("Augmentation completed!")
print(f"Original training images: 40")
print(f"Augmented training images: {len(os.listdir(output_images_dir))}")