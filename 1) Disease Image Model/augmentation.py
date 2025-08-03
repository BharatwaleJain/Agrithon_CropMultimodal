import albumentations as A
import cv2
import os
import shutil
def read_yolo_segmentation(label_path, img_height, img_width):
    polygons = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    pixel_coords = []
                    for i in range(0, len(coords), 2):
                        x = coords[i] * img_width
                        y = coords[i + 1] * img_height
                        pixel_coords.extend([x, y])
                    polygons.append([pixel_coords, class_id])
    return polygons
def write_yolo_segmentation(polygons, label_path, img_height, img_width):
    with open(label_path, 'w') as f:
        for polygon_data in polygons:
            if len(polygon_data) >= 2:
                coords, class_id = polygon_data
                normalized_coords = []
                for i in range(0, len(coords), 2):
                    x_norm = coords[i] / img_width
                    y_norm = coords[i + 1] / img_height
                    normalized_coords.extend([x_norm, y_norm])
                coords_str = ' '.join([f'{coord:.6f}' for coord in normalized_coords])
                f.write(f"{int(class_id)} {coords_str}\n")
transform = A.Compose([
    A.Rotate(limit=30, p=0.8, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
])
input_images_dir = 'yolov8_disease_dataset/images/train'
input_labels_dir = 'yolov8_disease_dataset/labels/train'
output_images_dir = 'augmented_disease_dataset/images/train'
output_labels_dir = 'augmented_disease_dataset/labels/train'
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)
os.makedirs('augmented_disease_dataset/images/val', exist_ok=True)
os.makedirs('augmented_disease_dataset/labels/val', exist_ok=True)
for img_name in os.listdir(input_images_dir):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        shutil.copy(os.path.join(input_images_dir, img_name), output_images_dir)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        if os.path.exists(os.path.join(input_labels_dir, label_name)):
            shutil.copy(os.path.join(input_labels_dir, label_name), output_labels_dir)
shutil.copytree('yolov8_disease_dataset/images/val', 'augmented_disease_dataset/images/val', dirs_exist_ok=True)
shutil.copytree('yolov8_disease_dataset/labels/val', 'augmented_disease_dataset/labels/val', dirs_exist_ok=True)
num_augmentations = 2
for img_name in os.listdir(input_images_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(input_images_dir, img_name)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(input_labels_dir, label_name)
    image = cv2.imread(img_path)
    img_height, img_width = image.shape[:2]
    polygons = read_yolo_segmentation(label_path, img_height, img_width)
    for aug_idx in range(num_augmentations):
        try:
            augmented = transform(image=image)
            aug_image = augmented['image']
            aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{aug_idx}.jpg"
            aug_img_path = os.path.join(output_images_dir, aug_img_name)
            cv2.imwrite(aug_img_path, aug_image)
            aug_label_name = f"{os.path.splitext(img_name)[0]}_aug_{aug_idx}.txt"
            aug_label_path = os.path.join(output_labels_dir, aug_label_name)
            write_yolo_segmentation(polygons, aug_label_path, aug_image.shape[0], aug_image.shape[1])
        except Exception as e:
            print(f"Error augmenting {img_name}: {e}")
print("Disease dataset augmentation completed!")