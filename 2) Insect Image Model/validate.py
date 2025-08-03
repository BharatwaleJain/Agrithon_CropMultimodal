from ultralytics import YOLO
import os
from pathlib import Path
model_path = 'runs/detect/train/weights/best.pt'
if not os.path.exists(model_path):
    print(f"Model not found at: {model_path}")
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        print("\nAvailable directories in runs/detect/:")
        for item in runs_dir.iterdir():
            if item.is_dir():
                weights_path = item / 'weights' / 'best.pt'
                if weights_path.exists():
                    print(f"Found model at: {weights_path}")
                    model_path = str(weights_path)
                    break
                else:
                    print(f"Directory: {item.name}")
    exit()
print(f"Model found at: {model_path}")
try:
    model = YOLO(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
print("\nValidating model...")
try:
    dataset_yaml = 'augmented_insect_dataset.yaml'
    if not os.path.exists(dataset_yaml):
        print(f"Dataset YAML not found: {dataset_yaml}")
        exit()
    metrics = model.val(data=dataset_yaml)
    print("\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
except Exception as e:
    print(f"Error during validation: {e}")