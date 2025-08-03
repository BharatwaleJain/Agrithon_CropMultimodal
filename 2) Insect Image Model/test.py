from ultralytics import YOLO
import os
from pathlib import Path
model_path = 'runs/detect/train/weights/best.pt'
if not os.path.exists(model_path):
    print(f"Model not found at: {model_path}")
    exit()
try:
    model = YOLO(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
print("\nTesting predictions...")
val_img_dir = 'augmented_insect_dataset/images/val'
if os.path.exists(val_img_dir):
    val_images = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if val_images:
        test_img_path = os.path.join(val_img_dir, val_images[0])
        print(f"Testing on: {test_img_path}")
        try:
            results = model.predict(test_img_path, conf=0.5, save=True)
            for i, result in enumerate(results):
                print(f"Prediction {i+1} completed")
                try:
                    result.show()
                except:
                    print("Cannot display image (non-GUI environment)")
                result.save()
                if hasattr(result, 'boxes') and result.boxes is not None:
                    num_detections = len(result.boxes)
                    print(f"Detected {num_detections} insects")
                    if num_detections > 0:
                        confidences = result.boxes.conf.cpu().numpy()
                        print(f"Confidence scores: {[f'{conf:.2f}' for conf in confidences]}")
                else:
                    print("No insects detected")
            print(f"Results saved to: runs/detect/predict/")
        except Exception as e:
            print(f"Error during prediction: {e}")
        print("\nTesting on multiple validation images...")
        try:
            test_images = val_images[:5] if len(val_images) >= 5 else val_images
            test_paths = [os.path.join(val_img_dir, img) for img in test_images]
            batch_results = model.predict(test_paths, conf=0.5, save=True)
            print(f"Batch prediction completed on {len(test_images)} images")
            print(f"All results saved to: runs/detect/predict/")
            total_detections = 0
            for i, result in enumerate(batch_results):
                if hasattr(result, 'boxes') and result.boxes is not None:
                    num_detections = len(result.boxes)
                    total_detections += num_detections
                    print(f"Image {i+1}: {num_detections} insects detected")
                else:
                    print(f"Image {i+1}: No insects detected")
            print(f"\nTotal detections across all test images: {total_detections}")
        except Exception as e:
            print(f"Error during batch prediction: {e}")
    else:
        print("No validation images found")
else:
    print(f"Validation directory not found: {val_img_dir}")